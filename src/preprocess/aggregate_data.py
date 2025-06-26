
import re
import pandas as pd

import numpy as np
from functools import reduce

from src.utils.varnames import ColNames as c
from config.config import (
    PROCESSED_DATA_PATH)
from src.utils.utils import weighted_mean_column



ps = pd.read_csv(PROCESSED_DATA_PATH / 'processed_sales.csv')

ps = ps[[c.DATE, c.SKU, c.SKU_PLATFORM, c.PLATFORM, c.SUPPLIER, c.LICENSE, c.UNITS, c.SALES_MXN, c.COST,  c.PRODUCT, c.CLUSTER]]
ps['ts_index'] = c.SKU_PLATFORM

meta = ps[[c.SKU_PLATFORM, c.CLUSTER, c.LICENSE]].drop_duplicates()

S_sku = pd.DataFrame(np.eye(len(meta)), index=meta[c.SKU_PLATFORM], columns=meta[c.SKU_PLATFORM])


S_cluster = pd.crosstab(meta[c.CLUSTER], meta[c.SKU_PLATFORM]).astype(int)

def aggregate_values(S: pd.DataFrame, value_cols: list):
    aggregation_level = S.index.name
    long_agg_dfs = []
    for value_col in value_cols:
        ps_pivot = ps.pivot(index=c.SKU_PLATFORM, columns=c.DATE, values=value_col).fillna(0)
        mmult = S @ ps_pivot
        mmult_long = mmult.reset_index().melt(
        id_vars=aggregation_level,
        var_name=c.DATE,
        value_name=value_col            # The time series values
    )
        long_agg_dfs.append(mmult_long)
    return merge_multiple_dataframes(long_agg_dfs, [aggregation_level, c.DATE])

def aggregate_rows(S: pd.DataFrame, value_cols: list, categorical_cols: list, average_cols: list) -> pd.DataFrame:
    aggregation_level = S.index.name
    values = aggregate_values(S, value_cols)
    count = ps.groupby([aggregation_level, c.DATE]).size().reset_index(name='count')
    values = values.merge(count, on=[aggregation_level, c.DATE], how='left')
    values = values.dropna(subset='count')
    categorical = ps[[aggregation_level] +  categorical_cols].drop_duplicates() #c.SUPPLIER, c.LICENSE, c.PRODUCT
    averages = ps[[aggregation_level, c.UNITS] + average_cols].copy()
    for average_col in average_cols:
        averages[average_col] = weighted_mean_column(averages, [aggregation_level], average_col)
    averages = averages.drop(c.UNITS, axis=1).drop_duplicates()
    aggregated = merge_multiple_dataframes([values, categorical, averages], aggregation_level)
    aggregated['ts_index'] = aggregated[aggregation_level]
    return aggregated



cluster = aggregate_rows(S_cluster, [c.UNITS, c.SALES_MXN], [c.SUPPLIER, c.LICENSE, c.PRODUCT], [c.COST])

# TODO  singleton_cluster = cluster.groupby(['ts_index'])['count'].mean().reset_index() eliminate singleton when reconciling

S_license = pd.crosstab(meta[c.LICENSE], meta[c.SKU_PLATFORM]).astype(int)


license = aggregate_rows(S_license, [c.UNITS, c.SALES_MXN], [], [c.COST])

data = pd.concat([ps, cluster, license], axis=0)


