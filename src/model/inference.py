import json
import re

import numpy as np
import pandas as pd
from catboost import Pool, CatBoostRegressor

from config.config import PROCESSED_DATA_PATH, MODELS_PATH, MODEL_DIR
from src.model.common import prepare_dataset, remove_singleton_aggregated_levels
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
    df = df.sort_values(by=['ts_index', c.DATE, 'cont_sales_id']).reset_index(drop=True)
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

def create_inference_pool(inf_i, features, categorical_features):
    categorical_features_t = [x for x in categorical_features if x in features]# .copy()
    inf_pool_i = Pool(data=inf_i[features], cat_features=categorical_features_t)
    return inf_pool_i

def inference(models_path, df, config):
    # Extract config
    categorical_features = config["categorical_features"]
    features_exclude = config["features_exclude"]
    nodes = config["aggregation_levels"]
    inference_date_cutoff = config["train_cutoff"]
    model_dir = config["model_dir"]

    df = prepare_dataset(df, inference_date_cutoff, categorical_features, "inference")
    features = [x for x in df.columns if x not in features_exclude]
    inference_dates = df[df[c.DATE] >= inference_date_cutoff][c.DATE].unique().tolist()
    lags, trends, rolls = extract_lag_related_feature_names(features)
    inf_nodes = []
    for node in nodes:
        model = CatBoostRegressor()
        model.load_model(models_path / model_dir / f"{node}_model.cbm")
        train_inf = df[(df[c.DATE] < inference_date_cutoff) & (df[c.NODE] == node)].copy()
        inf_i_lst = []
        for date in inference_dates:
            train_inf = pd.concat(
                [train_inf, df[(df[c.DATE] == date) & (df[c.NODE]== node)]], ignore_index=True)
            inf_i = create_inference_dataset(train_inf, lags, rolls, trends, date)
            inf_pool_i = create_inference_pool(inf_i, features, categorical_features)
            preds_i = model.predict(data=inf_pool_i)
            inf_i[c.UNITS] = preds_i
            inf_i_lst.append(inf_i)
        inf = pd.concat(inf_i_lst, ignore_index=True)
        inf_nodes.append(inf[[c.DATE, c.UNIQUE_ID, c.UNITS, 'count', c.CLUSTER, c.LICENSE]])
    return inf_nodes


def create_inference_dataset(df, lags, rolls, trends, date):
    df = compute_lag_related_features(df, lags, rolls, trends)
    inf_i = df[df[c.DATE] == date]
    inf_i = remove_singleton_aggregated_levels(inf_i)
    return inf_i


if __name__ == '__main__':
    
    data_features = pd.read_csv(PROCESSED_DATA_PATH / 'data_features.csv')
    MODEL_PATH = MODELS_PATH / MODEL_DIR
    
    with open(MODEL_PATH / "config.json", "r") as f:
        CONFIG = json.load(f)
    inference_nodes = inference(MODELS_PATH, data_features, CONFIG)
    
