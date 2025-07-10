import json
import importlib
from typing import Any

import numpy as np
import pandas as pd
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp
from pandas import DataFrame

from config.config import PROCESSED_DATA_PATH, MODELS_PATH, MODEL_DIR
from src.model.inference import inference
from src.utils.varnames import ColNames as c



def concatenate_nodes_build_s_and_tags(fcst_nodes, nodes):
    tags = {}
    s_nodes = []
    updated_fcst_nodes = []
    for i in reversed(range(len(nodes))):
        node = nodes[i]
        fcst_node = fcst_nodes[i]
        if node != 'bottom':
            updated_df = restore_singleton_parents(fcst_node, fcst_nodes, node)
            s_node = build_parents_summing_matrix(fcst_nodes[0], node)
        else:
            updated_df = fcst_node
            children_ids = s_nodes[i + 1].columns.values
            s_node = build_children_summation_matrix(children_ids)
        updated_df[c.NODE] = node
        updated_fcst_nodes.append(updated_df)
        s_nodes.append(s_node)
        tags[node] = updated_df[c.UNIQUE_ID].unique()
    return (pd.concat(updated_fcst_nodes, ignore_index=True),
            pd.concat(s_nodes, ignore_index=False).reset_index(names=c.UNIQUE_ID), tags)


def build_children_summation_matrix(children_ids):
    s_node = pd.DataFrame(
        np.eye(len(children_ids)),
        index=children_ids,
        columns=children_ids
    )
    return s_node


def build_parents_summing_matrix(child_fcst, node):
    meta = child_fcst[[c.UNIQUE_ID, node]].drop_duplicates()
    s_node = pd.crosstab(meta[node], meta[c.UNIQUE_ID]).astype(int)
    return s_node


def restore_singleton_parents(fcst_node, fcst_nodes, node):
    singletons = retrieve_singleton_nodes(fcst_nodes, node)
    updated_df = pd.concat([fcst_node, singletons])
    return updated_df


def load_class(class_path: str):
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def retrieve_singleton_nodes(fcst_nodes, node):
    child_counts = fcst_nodes[0].groupby([node, c.DATE])[c.UNIQUE_ID].nunique().reset_index(
        name='n_children')
    singleton_parents = child_counts[child_counts['n_children'] == 1]
    singletons = fcst_nodes[0].merge(singleton_parents[[node, c.DATE]], on=[node, c.DATE])
    singletons[c.UNIQUE_ID] = singletons[node]
    return singletons

def fill_dates_for_inconsistent_series(fcst, model_name, target):
    pivot = fcst.pivot(index=c.UNIQUE_ID, columns=c.DATE, values=target)
    long_format = pivot.reset_index().melt(id_vars=c.UNIQUE_ID, var_name=c.DATE, value_name=target).sort_values(
        [c.UNIQUE_ID, c.DATE]).reset_index(drop=True)
    long_format[model_name] = long_format[target].fillna(0)
    long_format.drop(target, axis=1, inplace=True)
    return long_format


def round_forecasts_and_reconcile_bottom_up(fcst, summ_matrix, tags):
    numeric_cols = fcst.select_dtypes(include='number').columns
    fcst[numeric_cols] = fcst[numeric_cols].round()
    hrec = HierarchicalReconciliation(reconcilers=[BottomUp()])
    fcst_reconciled = hrec.reconcile(Y_hat_df=fcst, S=summ_matrix, tags=tags, id_col=c.UNIQUE_ID, time_col=c.DATE)
    fcst_reconciled[c.DATE] = pd.to_datetime(fcst_reconciled[c.DATE]).dt.strftime('%Y-%m-%d')
    return fcst_reconciled.iloc[:, [0, 1, 4, 5]]


def reconciliation(models_path, dfs: pd.DataFrame, config: dict):
    nodes = config["aggregation_levels"]
    model_name: object = config["model_dir"]
    target = config["target"]
    reconcilers_config = config["reconcilers"]

    concat_nodes, S_df, tags = concatenate_nodes_build_s_and_tags(dfs, nodes)
    Y_hat = fill_dates_for_inconsistent_series(concat_nodes, model_name, target)
    reconcilers = [
        load_class(rec_method["class_path"])(**rec_method["params"])
        for rec_method in reconcilers_config
    ]
    if len(reconcilers) > 0:
        hrec = HierarchicalReconciliation(reconcilers=reconcilers)
        Y_hat = hrec.reconcile(Y_hat_df=Y_hat, S=S_df, tags=tags, id_col=c.UNIQUE_ID, time_col=c.DATE)
    Y_hat = round_forecasts_and_reconcile_bottom_up(Y_hat, S_df, tags)
    Y_hat = Y_hat.merge(concat_nodes[[c.UNIQUE_ID, c.NODE]].drop_duplicates(), on=c.UNIQUE_ID)
    Y_hat.to_csv(models_path / model_name / f"inference.csv", index=False)




if __name__ == '__main__':
    data_features = pd.read_csv(PROCESSED_DATA_PATH / 'data_features.csv')
    MODEL_PATH = MODELS_PATH / MODEL_DIR

    with open(MODEL_PATH / "config.json", "r") as f:
        CONFIG = json.load(f)

    # Extract config
    inference_nodes = inference(MODELS_PATH, data_features, CONFIG)
    reconciliation(MODELS_PATH, inference_nodes, CONFIG)