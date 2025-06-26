from functools import reduce

import pandas as pd

from src.utils.varnames import ColNames as c


def merge_sales_and_imdb_data(dataframes: list[pd.DataFrame], merge_key: str, important_licenses) -> pd.DataFrame:
    df = reduce(lambda left, right: pd.merge(left, right, on=merge_key,
                                               how='left'), dataframes)
    df = get_months_from_release(df, important_licenses)
    return df

def get_months_from_release(df: pd.DataFrame, important_licenses) -> pd.DataFrame:
    df[c.DATE] = pd.to_datetime(df[c.DATE], format="%Y-%m-%d")
    df["release_date"] = pd.to_datetime(df["release_date"], format="%m/%d/%y")
    df['months_from_release'] = (
            (df[c.DATE].dt.year - df['release_date'].dt.year) * 12 +
            (df[c.DATE].dt.month - df['release_date'].dt.month)
    )
    dummies = pd.get_dummies(df[c.LICENSE])[important_licenses].astype(int).multiply(
        df["months_from_release"], axis=0)
    dummies.columns = [f'months_from_release_{(cat.replace(" ", "_").lower())}' for cat in important_licenses]
    for col in dummies.columns:
        df[col] = dummies[col].fillna(0)
    return df