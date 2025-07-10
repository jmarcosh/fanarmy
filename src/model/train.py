import pandas as pd
import json
from catboost import CatBoostRegressor, Pool

from src.model.common import prepare_dataset, get_fixed_prices_per_series
from src.utils.varnames import ColNames as c
from config.config import (
    PROCESSED_DATA_PATH, MODELS_PATH, MODEL_DIR, AGGREGATION_LEVELS)




def create_train_pool(df, node, price_per_sku, features, categorical_features, target):
    train_df = df[(df["aggregation_level"] == node)].copy()
    train_df[c.PRICE] = train_df['ts_index'].map(price_per_sku)
    train_df = remove_singleton_aggregated_levels(train_df)
    categorical_features_t = [x for x in categorical_features if x in features]
    train_pool = Pool(train_df[features], label=train_df[target], cat_features=categorical_features_t)
    return train_pool


def remove_singleton_aggregated_levels(train_df):
    singleton_agg = train_df.groupby(['ts_index', 'aggregation_level'])['count'].mean().reset_index()
    singleton_agg_index = singleton_agg.loc[
        (singleton_agg['aggregation_level'] != 'bottom') & (singleton_agg['count'] == 1), 'ts_index'].tolist()
    train_df = train_df[~(train_df['ts_index'].isin(singleton_agg_index))]
    return train_df



def train(df, train_date, categorical_features, features_drop, nodes, params, target, model_path):

    df = prepare_dataset(df, train_date, categorical_features, "train")
    features = [x for x in df.columns if x not in features_drop]
    price_per_sku = get_fixed_prices_per_series(df)

    for node in nodes:
        train_pool = create_train_pool(df, node, price_per_sku, features, categorical_features, target)
        model = CatBoostRegressor(**params, verbose=0)
        model.fit(train_pool)
        model.save_model(model_path / f"{node}_model.cbm")


if __name__ == '__main__':
    data_features = pd.read_csv(PROCESSED_DATA_PATH / 'data_features.csv')
    MODEL_PATH = MODELS_PATH / MODEL_DIR
    with open(MODEL_PATH / "config.json", "r") as f:
        config = json.load(f)

    # Extract config
    PARAMS = config["params"]
    FEATURES_EXCLUDE = config["features_exclude"]
    CATEGORICAL_FEATURES = config["categorical_features"]
    AGGREGATION_LEVELS = config["aggregation_levels"]
    TARGET = config["target"]
    TRAIN_CUTOFF = config["train_cutoff"]


    train(data_features, TRAIN_CUTOFF, CATEGORICAL_FEATURES, FEATURES_EXCLUDE, AGGREGATION_LEVELS,
          PARAMS, TARGET, MODEL_PATH)
