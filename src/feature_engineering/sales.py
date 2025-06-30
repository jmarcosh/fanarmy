import numpy as np
import pandas as pd

from src.utils.varnames import ColNames as c


def engineer_sales_features(df: pd.DataFrame, periods_in_year:int, max_lag: int, valid_lags_threshold: float) -> pd.DataFrame:
    df = extract_year_and_month_from_date(df)
    df = get_continuous_sales_id(df)
    df = engineer_time_features(df, periods_in_year)
    df = get_sales_lags(df, max_lag, valid_lags_threshold)
    return df

def engineer_time_features(df: pd.DataFrame, periods_in_year: int) -> pd.DataFrame:
    df['month_sin'] = np.sin(2 * np.pi * df[c.MONTH_NUM] / periods_in_year)
    df['month_cos'] = np.cos(2 * np.pi * df[c.MONTH_NUM] / periods_in_year)
    df['months_on_shelf'] = (
        df.groupby(['ts_index', 'cont_sales_id'], observed=True)
        .cumcount()
    )
    return df.drop([c.MONTH_NUM], axis=1)

def count_valid_lags(df, sku_cols, date_col, target_col, max_lag=12):
    df = df.copy()
    df = df.sort_values(sku_cols + [date_col])

    lag_counts = {}

    for lag in range(1, max_lag + 1):
        df[f'lag_{lag}'] = df.groupby(sku_cols, observed=True)[target_col].shift(lag)
        # Count non-NaNs across all SKUs
        lag_counts[lag] = df[f'lag_{lag}'].notna().sum()

    return pd.Series(lag_counts)


def get_sales_lags(df: pd.DataFrame, max_lag, valid_lags_threshold: float) -> pd.DataFrame: # , correlation_threshold: float
    #TODO calculate max_lag by substracting min_date from max_date
    lag_valid_counts = count_valid_lags(df, sku_cols=['ts_index', 'cont_sales_id'], date_col=c.DATE,
                                        target_col=c.UNITS, max_lag=max_lag)
    threshold_cnt = valid_lags_threshold * len(df)
    selected_lags = lag_valid_counts[lag_valid_counts >= threshold_cnt].index.tolist()
    for lag in selected_lags:
        df[f'lag_{lag}'] = df.groupby(['ts_index', 'cont_sales_id'], observed=True)[c.UNITS].shift(lag)
    correlations = {
        f'lag_{lag}': df[[f'lag_{lag}', c.UNITS]].corr().iloc[0, 1]
        for lag in selected_lags
    }
    # We will prune features after training model
    # drop_keys = [k for k, v in correlations.items() if v < correlation_threshold]
    # df = df.drop(drop_keys, axis=1)
    df = get_sales_trends(df, correlations)
    df['sales_roll_3'] = df.groupby(['ts_index', 'cont_sales_id'], observed=True)[c.UNITS].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
    return df

def get_sales_trends(df: pd.DataFrame, correlations) -> pd.DataFrame:
    keys = list(correlations.keys())
    # First trend lag_1 / lag_2
    if len(correlations) > 2:
        df[f'{keys[0]}_over_{keys[1]}'] = df[f'{keys[0]}'] / df[f'{keys[1]}']

    # Step 2: Find the key with the highest value
    first_max_key = max(keys, key=lambda k: correlations[k])
    first_max_index = keys.index(first_max_key)

    # Step 3: Look at the keys after the first max
    remaining_keys = keys[first_max_index + 1:]

    # Step 4: Find the second highest if any keys remain
    if remaining_keys:
        second_max_key = max(remaining_keys, key=lambda k: correlations[k])
        df[f'{first_max_key}_over_{second_max_key}'] = df[f'{first_max_key}'] / df[f'{second_max_key}']
    return df

def get_continuous_sales_id(df):
    df['prev_date'] = df.groupby('ts_index')[c.DATE].shift(1)
    df['month_diff'] = (
            (df[c.YEAR] - df['prev_date'].dt.year) * 12 +
            (df[c.MONTH_NUM] - df['prev_date'].dt.month)
    )

    # Mark where continuity breaks (i.e., not exactly 1 month apart or first row)
    df['new_sequence'] = (df['month_diff'] != 1) | df['month_diff'].isna()

    # Cumulative count of breaks within each group gives the sequence ID
    df['cont_sales_id'] = df.groupby('ts_index')['new_sequence'].cumsum()
    return df.drop(['prev_date', 'month_diff', 'new_sequence'], axis=1)

def extract_year_and_month_from_date(df):
    df[c.DATE] = pd.to_datetime(df[c.DATE])
    df[c.YEAR] = df[c.DATE].dt.year
    df[c.MONTH_NUM] = df[c.DATE].dt.month
    return df