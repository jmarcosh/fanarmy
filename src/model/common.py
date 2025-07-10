import numpy as np
import pandas as pd
from src.utils.varnames import ColNames as c


def get_fixed_prices_per_series(df):
    ts_index_sales = df.groupby('ts_index')[[c.SALES_MXN, c.UNITS]].sum()
    return (ts_index_sales[c.SALES_MXN] / ts_index_sales[c.UNITS]).replace([np.inf, -np.inf], np.nan).rename(
        c.PRICE)  # Fixed price over time. We won't have the exact price for inference


def prepare_dataset(df, date_cutoff, categorical_features, mode):
    df[c.DATE] = pd.to_datetime(df[c.DATE], format="%Y-%m-%d")
    df = df.replace([np.inf, -np.inf], np.nan)
    # df[c.CLUSTER] = pd.Categorical(df[c.CLUSTER])
    for col in categorical_features:
        df[col] = df[col].astype(str)
    if mode == 'train':
        df = df[df[c.DATE] < date_cutoff]
    elif mode == 'inference':
        df.loc[df[c.DATE] >= date_cutoff, c.UNITS] = np.nan
    price_per_sku = get_fixed_prices_per_series(df) # make sure after removing test set to avoid leakage
    df[c.PRICE] = df[c.UNIQUE_ID].map(price_per_sku)
    return df

def remove_singleton_aggregated_levels(df):
    return df[(df[c.NODE] == 'bottom') | (df['count'] != 1)]