import numpy as np
import pandas as pd

from src.utils.varnames import ColNames as c

def engineer_moviemeter_features(df, moviemeter_trend_windows) -> pd.DataFrame:
    df['moviemeter_average'] = df.iloc[:, 1:].mean(axis=1)
    for window in moviemeter_trend_windows:
        df[f'moviemeter_trend_{window}'] = get_moviemeter_trends(df, window)
    return df[['license', 'moviemeter_average', 'moviemeter_trend_3', 'moviemeter_trend_6']].rename(
        {'license': c.LICENSE}, axis=1)

def get_moviemeter_trends(df: pd.DataFrame, window: int) -> pd.Series:
    if df.shape[1] < 1 + 2 * window:
        raise ValueError("DataFrame does not have enough columns for the specified window size.")

    first_cols = df.columns[1: 1 + window]
    last_cols = df.columns[-window:]

    first_sum = df[first_cols].sum(axis=1)
    last_sum = df[last_cols].sum(axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        trend = last_sum / first_sum
        trend = trend.replace([np.inf, -np.inf], np.nan)

    return trend