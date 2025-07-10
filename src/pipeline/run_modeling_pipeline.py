import json

import pandas as pd

from config.config import (PROCESSED_DATA_PATH, MODELS_PATH, MODEL_DIR, TRAIN)
from src.model.reconciliation import reconciliation

from src.model.save_config import save_model_config_to_json
from src.model.train import train
from src.model.inference import inference


def run_modeling_pipeline():
    save_model_config_to_json(MODELS_PATH)

    df = pd.read_csv(PROCESSED_DATA_PATH / 'data_features.csv')
    with open(MODELS_PATH / MODEL_DIR / "config.json", "r") as f:
        config = json.load(f)
    if TRAIN:
        train(MODELS_PATH, df, config)

    inference_nodes = inference(MODELS_PATH, df, config)
    reconciliation(MODELS_PATH, inference_nodes, config)

if __name__ == '__main__':
    run_modeling_pipeline()