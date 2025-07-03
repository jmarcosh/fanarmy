import numpy as np
import pandas as pd
from src.utils.varnames import ColNames as c


def get_fixed_prices_per_series(df):
    ts_index_sales = df.groupby('ts_index')[[c.SALES_MXN, c.UNITS]].sum()
    return (ts_index_sales[c.SALES_MXN] / ts_index_sales[c.UNITS]).rename(
        c.PRICE)  # Fixed price over time. We won't have the exact price for inference


def prepare_dataset(df, date_cutoff, categorical_features, mode):
    df[c.DATE] = pd.to_datetime(df[c.DATE], format="%Y-%m-%d")
    # df[c.CLUSTER] = pd.Categorical(df[c.CLUSTER])
    for col in categorical_features:
        df[col] = df[col].astype(str)
    if mode == 'train':
        df = df[df[c.DATE] < date_cutoff]
    return df