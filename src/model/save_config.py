import json
from config.config import (PARAMS, FEATURES_EXCLUDE, CATEGORICAL_FEATURES, AGGREGATION_LEVELS, TARGET, TRAIN_CUTOFF,
                           MODELS_PATH, MODEL_DIR
)

config_dict = {"params": PARAMS,
               "features_exclude": FEATURES_EXCLUDE,
               "categorical_features": CATEGORICAL_FEATURES,
               "aggregation_levels": ["bottom"] + AGGREGATION_LEVELS,
               "target": TARGET,
               "train_cutoff": TRAIN_CUTOFF,
               "model_dir": MODEL_DIR,
}

with open(f"{MODELS_PATH / MODEL_DIR}/config.json", "w") as f:
    json.dump(config_dict, f, indent=4)