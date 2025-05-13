import itertools
import pandas as pd

from src.preprocess_data.utils.varnames import MONTH_NAME_TO_NUMBER, ColNames


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
    return result


def load_sales_data(data_path, platform_include, supplier_exclude):
    c = ColNames()
    df = pd.read_excel(data_path, sheet_name='Data')
    df = df[(~df[c.SUPPLIER].isin(supplier_exclude)) & (df[c.PLATFORM].isin(platform_include))].reset_index(
        drop=True)
    df = df.groupby([
        c.SKU, c.DESCRIPTION, c.SUPPLIER, c.LICENSE, c.PLATFORM, c.MONTH, c.YEAR
    ])[[c.UNITS, c.SALES_MXN, c.COST]].sum().reset_index()
    # Add ate in yyyy/mm/dd format
    df[c.MONTH_NUM] = df[c.MONTH].str.lower().map(MONTH_NAME_TO_NUMBER)
    df[c.DATE] = pd.to_datetime(dict(year=df[c.YEAR], month=df[c.MONTH_NUM], day=1))
    # Expand data with months we have no sales observations
    df = expand_df_with_all_combinations(df, 'MS', [c.SKU, c.DESCRIPTION, c.SUPPLIER, c.LICENSE, c.PLATFORM],
                                            date_col=c.DATE, platform_col=c.PLATFORM)
    df = df.groupby([c.SKU, c.DESCRIPTION, c.SUPPLIER,
                           c.LICENSE, c.PLATFORM]).filter(lambda g: g[c.UNITS].notna().any())
    df[c.UNITS] = df[c.UNITS].fillna(0)
    return df