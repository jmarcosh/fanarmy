from functools import reduce
import pandas as pd
import numpy as np
from src.utils.varnames import ColNames as c

def weighted_mean(x, weight_series):
    x = x.dropna()
    if x.empty:
        return np.nan
    w = weight_series.loc[x.index]
    return np.average(x, weights=w)

def weighted_mean_column(df, groupby_cols, series):
    return df.groupby(groupby_cols)[series].transform(
    lambda x: weighted_mean(x, df[c.UNITS]))

def merge_multiple_dataframes(dfs: list, on: list|str) -> pd.DataFrame:
    return reduce(lambda  left,right: pd.merge(left,right,on=on,
                                            how='outer'), dfs)