import itertools

import numpy as np
import pandas as pd

from src.utils.utils import weighted_mean_column
from src.utils.varnames import MONTH_NAME_TO_NUMBER, ColNames as c


def expand_df_with_all_combinations(df, freq, grouping_cols, date_col='sale_date', platform_col=''):
    # Compute min and max dates for each group
    if len(platform_col) > 0:
        range_grouping_cols = [col for col in grouping_cols if col != platform_col]
        date_ranges_base = df.groupby(range_grouping_cols)[date_col].agg(['min', 'max']).reset_index()
        date_ranges_lst = []
        for platform in df[platform_col].unique():
            date_ranges_temp = date_ranges_base.copy()
            date_ranges_temp[platform_col] = platform
            date_ranges_lst.append(date_ranges_temp)

        date_ranges = pd.concat(date_ranges_lst)
    else:
        date_ranges = df.groupby(grouping_cols)[date_col].agg(['min', 'max']).reset_index()
    # Create a list to collect all group-date combinations
    all_combinations = []

    for _, row in date_ranges.iterrows():
        # Build date range for this group
        dates = pd.date_range(start=row['min'], end=row['max'], freq=freq)

        # Extract the grouping column values
        group_values = row[grouping_cols].values

        # Combine dates with group values
        group_combinations = pd.DataFrame(
            itertools.product(dates, [group_values]),
            columns=[date_col, 'group_comb']
        )

        # Expand group values into columns
        group_cols_df = pd.DataFrame(group_combinations['group_comb'].tolist(), columns=grouping_cols)
        combined = pd.concat([group_combinations[[date_col]], group_cols_df], axis=1)

        all_combinations.append(combined)

    # Concatenate all group-date combinations
    full_combinations_df = pd.concat(all_combinations, ignore_index=True)

    # Merge with original data
    result = full_combinations_df.merge(df, on=[date_col] + grouping_cols, how='left')

    # Sort and return
    result = result.sort_values(by=grouping_cols + [date_col]).reset_index(drop=True)
    
    result = result.groupby([c.SKU, c.DESCRIPTION, c.SUPPLIER,
                           c.LICENSE, c.PLATFORM]).filter(lambda g: g[c.UNITS].notna().any())
    return result


def load_sales_data(data_path, platform_include, supplier_exclude):
    df = pd.read_excel(data_path, sheet_name='Data')
    df = df[(~df[c.SUPPLIER].isin(supplier_exclude)) & (df[c.PLATFORM].isin(platform_include))].reset_index(
        drop=True)
    df = df.groupby([
        c.SKU, c.DESCRIPTION, c.SUPPLIER, c.LICENSE, c.PLATFORM, c.MONTH, c.YEAR
    ]).agg({c.UNITS: 'sum', c.SALES_MXN: 'sum', c.COST: 'mean'}).reset_index()
    # Add ate in yyyy/mm/dd format
    for col in df.select_dtypes(include=['string', 'object']).columns:
        df[col] = df[col].astype(str).str.strip()
    df[c.MONTH_NUM] = df[c.MONTH].str.lower().map(MONTH_NAME_TO_NUMBER)
    df[c.DATE] = pd.to_datetime(dict(year=df[c.YEAR], month=df[c.MONTH_NUM], day=1))
    # Expand data with months we have no sales observations
    df = expand_df_with_all_combinations(df, 'MS', [c.SKU, c.DESCRIPTION, c.SUPPLIER, c.LICENSE, c.PLATFORM],
                                            date_col=c.DATE, platform_col=c.PLATFORM)
    df[c.UNITS] = df[c.UNITS].fillna(0)
    df[c.SKU_PLATFORM] = df[c.SKU].astype(str) + "_" + df[c.PLATFORM].astype(str)
    df = add_price_and_cost(df)
    df = add_year_and_month(df)  
    df = check_for_missing_values(df, [c.SKU, c.DESCRIPTION, c.SUPPLIER,
           c.LICENSE, c.PLATFORM, c.PRICE])
    return df



def filter_out_skus_with_non_significant_sales(df, sales_threshold):
    data_sku = df.groupby([
        c.SKU], observed=True)[[c.UNITS]].sum().reset_index()
    sorted_sales = data_sku.sort_values(by=[c.UNITS], ascending=False)
    sorted_sales_array = sorted_sales[c.UNITS].values
    # % of SKUs
    sku_pct = np.arange(1, len(sorted_sales_array) + 1) / len(sorted_sales_array)
    # Cumulative sales as % of total
    cum_sales_pct = np.cumsum(sorted_sales_array) / sorted_sales_array.sum()
    # Find index of first value greater than the threshold
    index = np.argmax(sorted_sales_array == sales_threshold)
    selected_skus = sorted_sales[c.SKU].iloc[:index].tolist()
    dfs = df[df[c.SKU].isin(selected_skus)].reset_index(drop=True)
    print(f'Percentage of SKUs with sales above {sales_threshold}: {sku_pct[index]:.2%}')
    print(f'Percentage of rows kept: {len(dfs) / len(df):.2%}')
    print(f'Percentage of sales in SKUs above threshold: {cum_sales_pct[index]:.2%}')
    return dfs


def add_year_and_month(df):
    df[c.MONTH_NUM] = df[c.DATE].dt.month
    df[c.YEAR] = df[c.DATE].dt.year
    return df

def add_price_and_cost(df: pd.DataFrame) -> pd.DataFrame:
    df[c.PRICE] = df[c.SALES_MXN] / df[c.UNITS]
    df[c.PRICE] = df[c.PRICE].fillna(weighted_mean_column(df, [c.SKU_PLATFORM], c.PRICE))
    df[c.COST] = df[c.COST].fillna(
        weighted_mean_column(df, [c.SKU], c.COST)
    )
    return df

def check_for_missing_values(df: pd.DataFrame, required_columns) -> pd.DataFrame:
    if df[required_columns].isnull().any().any():
        raise ValueError("Missing values detected in required columns.")
    return df