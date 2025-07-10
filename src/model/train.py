import pandas as pd
import json
from catboost import CatBoostRegressor, Pool

from src.model.common import prepare_dataset, remove_singleton_aggregated_levels
from config.config import (
    PROCESSED_DATA_PATH, MODELS_PATH, MODEL_DIR)




def create_train_pool(df, node, features, categorical_features, target):
    train_df = df[(df["aggregation_level"] == node)].copy()
    train_df = remove_singleton_aggregated_levels(train_df)
    categorical_features_t = [x for x in categorical_features if x in features]
    train_pool = Pool(train_df[features], label=train_df[target], cat_features=categorical_features_t)
    return train_pool

def train(models_path, df, config):
    # Extract config
    params = config["params"]
    features_exclude = config["features_exclude"]
    categorical_features = config["categorical_features"]
    nodes = config["aggregation_levels"]
    target = config["target"]
    train_cutoff = config["train_cutoff"]
    model_dir = config["model_dir"]

    df = prepare_dataset(df, train_cutoff, categorical_features, "train")
    features = [x for x in df.columns if x not in features_exclude]
    for node in nodes:
        train_pool = create_train_pool(df, node, features, categorical_features, target)
        model = CatBoostRegressor(**params, verbose=0)
        model.fit(train_pool)
        model.save_model(models_path / model_dir /f"{node}_model.cbm")


if __name__ == '__main__':
    DATA_FEATURES = pd.read_csv(PROCESSED_DATA_PATH / 'data_features.csv')
    MODEL_PATH = MODELS_PATH / MODEL_DIR
    with open(MODEL_PATH / "config.json", "r") as f:
        CONFIG = json.load(f)



    train(MODELS_PATH, DATA_FEATURES, CONFIG)
