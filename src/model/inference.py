import json
import re
from dataclasses import asdict

import numpy as np
import pandas as pd
from catboost import Pool, CatBoostRegressor
from numpy.ma.testutils import assert_array_almost_equal

from config.config import PROCESSED_DATA_PATH, MODELS_PATH, MODEL_DIR
from src.model.common import prepare_dataset, get_fixed_prices_per_series
from src.utils.varnames import ColNames as c


def extract_lag_related_feature_names(features):
    lags = sorted({int(re.search(r'lag_(\d+)$', col).group(1))
                   for col in features if re.match(r'lag_\d+$', col)})
    # Extract trends (pairs of lags from pattern "lag_x_over_lag_y")
    trends = [[int(x), int(y)]
              for col in features
              if (match := re.match(r'lag_(\d+)_over_lag_(\d+)', col))
              for x, y in [match.groups()]]
    # Extract roll windows from "sales_roll_n"
    rolls = sorted({int(match.group(1))
                    for col in features
                    if (match := re.match(r'sales_roll_(\d+)', col))})
    return lags, trends, rolls

def compute_lag_related_features(df, lags, rolls, trends):
    df = df.sort_values(by=['ts_index', c.DATE, 'cont_sales_id'])
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby(['ts_index', 'cont_sales_id'], observed=True)[
            c.UNITS].shift(lag)
    for trend in trends:
        df[f'lag_{trend[0]}_over_lag_{trend[1]}'] = (
                df[f'lag_{trend[0]}'] / df[f'lag_{trend[1]}']).replace(
            [float('inf'), -float('inf')], np.nan)
    for roll in rolls:
        df[f'sales_roll_{roll}'] = df.groupby(['ts_index', 'cont_sales_id'], observed=True)[
            c.UNITS].transform(
            lambda x: x.shift(1).rolling(window=roll, min_periods=1).mean())
    return df

def create_inference_pool(df, price_per_sku, lags, rolls, trends,date, features, categorical_features):
    df[c.PRICE] = df['ts_index'].map(price_per_sku)
    df = compute_lag_related_features(df, lags, rolls, trends)
    inf_i = df[df[c.DATE] == date]
    categorical_features_t = [x for x in categorical_features if x in features]# .copy()
    inf_pool_i = Pool(data=inf_i[features], cat_features=categorical_features_t)
    return inf_i, inf_pool_i

def inference(df, inference_date_cutoff, categorical_features, features_drop, nodes, model_path):

    df = prepare_dataset(df, inference_date_cutoff, categorical_features, "inference")
    features = [x for x in df.columns if x not in features_drop]
    price_per_sku = get_fixed_prices_per_series(df)
    inference_dates = df[df[c.DATE] >= inference_date_cutoff][c.DATE].unique().tolist()
    lags, trends, rolls = extract_lag_related_feature_names(features)
    inf_nodes = []
    inf_nodes2 = []
    for node in nodes:
        time_step_preds = []
        time_step_inf_dfs = []
        model = CatBoostRegressor()
        model.load_model(model_path / f"{node}_model.cbm")
        train_inf = df[(df[c.DATE] < inference_date_cutoff) & (df["aggregation_level"] == node)].copy()
        for date in inference_dates:
            train_inf = pd.concat(
                [train_inf, df[(df[c.DATE] == date) & (df["aggregation_level"]== node)]])
            inf_i, inf_pool_i = create_inference_pool(train_inf, price_per_sku, lags, rolls, trends,date, features, categorical_features)
            preds_i = model.predict(data=inf_pool_i)
            time_step_preds += preds_i.tolist()
            time_step_inf_dfs.append(inf_i)
            train_inf.loc[(train_inf[c.DATE] == date), c.UNITS] = preds_i

        inf = pd.concat(time_step_inf_dfs)
        inf['predictions'] = time_step_preds
        inf2 = train_inf[(train_inf[c.DATE] >= inference_date_cutoff)]
        inf_nodes.append(inf)
        inf_nodes2.append(inf2)
        return inf_nodes



if __name__ == '__main__':
    
    data_features = pd.read_csv(PROCESSED_DATA_PATH / 'data_features.csv')
    MODEL_PATH = MODELS_PATH / MODEL_DIR
    
    with open(MODEL_PATH / "config.json", "r") as f:
        config = json.load(f)
    
    # Extract config
    CATEGORICAL_FEATURES = config["categorical_features"]
    FEATURES_EXCLUDE = config["features_exclude"]
    NODES = config["aggregation_levels"]
    TARGET = config["target"]
    TRAIN_CUTOFF = config["train_cutoff"]
    inference_nodes = inference(data_features, TRAIN_CUTOFF, CATEGORICAL_FEATURES, FEATURES_EXCLUDE, NODES, MODEL_PATH)
    
