import pandas as pd


from src.utils.varnames import ColNames as c
from config.config import (
    PROCESSED_DATA_PATH)
from src.utils.utils import weighted_mean_column, merge_multiple_dataframes


def aggregate_values(df, bottom_var, aggregation_level, value_cols: list):
    meta = df[[bottom_var, aggregation_level]].drop_duplicates()
    S = pd.crosstab(meta[aggregation_level], meta[c.SKU_PLATFORM]).astype(int)
    long_agg_dfs = []
    for value_col in value_cols:
        df_pivot = df.pivot(index=bottom_var, columns=c.DATE, values=value_col).fillna(0)
        mmult = S @ df_pivot
        mmult_long = mmult.reset_index().melt(
        id_vars=aggregation_level,
        var_name=c.DATE,
        value_name=value_col            # The time series values
    )
        mmult_long[c.DATE] = pd.to_datetime(mmult_long[c.DATE])
        long_agg_dfs.append(mmult_long)
    return merge_multiple_dataframes(long_agg_dfs, [aggregation_level, c.DATE])

def aggregate_rows(df, bottom_var, aggregation_level, value_cols: list, categorical_cols: list, average_cols: list) -> pd.DataFrame:
    values = aggregate_values(df, bottom_var, aggregation_level, value_cols)
    values = add_sku_count_per_aggregation_level(aggregation_level, df, values)
    categorical_cols = keep_unique_categorical_columns_per_aggregation_level(categorical_cols, aggregation_level, df)
    categorical = df[[aggregation_level] +  categorical_cols].drop_duplicates() #c.SUPPLIER, c.LICENSE, c.PRODUCT
    averages = df[[aggregation_level, c.UNITS] + average_cols].copy()
    for average_col in average_cols:
        averages[average_col] = weighted_mean_column(averages, [aggregation_level], average_col)
    averages = averages.drop(c.UNITS, axis=1).drop_duplicates()
    aggregated = merge_multiple_dataframes([values, categorical, averages], aggregation_level)
    # for var in [aggregation_level] +  categorical_cols:
    #     aggregated[var] = aggregation_level +"_" + aggregated[var].astype(str)
    aggregated['aggregation_level'] = aggregation_level
    aggregated['ts_index'] = aggregated[aggregation_level]
    return aggregated


def add_sku_count_per_aggregation_level(aggregation_level, df, values):
    count = df.groupby([aggregation_level, c.DATE]).size().reset_index(name='count')
    values = values.merge(count, on=[aggregation_level, c.DATE], how='left')
    values = values.dropna(subset='count')
    return values

def keep_unique_categorical_columns_per_aggregation_level(categorical_cols, aggregation_level, df):
    if aggregation_level in categorical_cols:
        categorical_cols.remove(aggregation_level)
    categorical_cols = [
        var for var in categorical_cols
        if (df.groupby(aggregation_level)[var].nunique() == 1).all()
    ]
    return categorical_cols




def aggregate_data(df, bottom_var, aggregation_levels, value_cols, categorical_cols, average_cols):
    df['count'] = 1
    df['aggregation_level'] = 'bottom'
    df['ts_index'] = df[bottom_var]
    # S_sku = pd.DataFrame(np.eye(len(meta)), index=meta[c.SKU_PLATFORM], columns=meta[c.SKU_PLATFORM]) # TODO when creating S
    agg_dfs = [df]
    for aggregation_level in aggregation_levels:
        agg_dfs.append(aggregate_rows(df, bottom_var, aggregation_level, value_cols,
                                    categorical_cols, average_cols))
    return pd.concat(agg_dfs, axis=0).reset_index(drop=True)


if __name__ == '__main__':
    BOTTOM_VAR = c.SKU_PLATFORM
    AGGREGATION_LEVELS = [c.CLUSTER, c.LICENSE]
    VALUE_COLS = [c.UNITS, c.SALES_MXN]
    CATEGORICAL_COLS = [c.SUPPLIER, c.LICENSE, c.PRODUCT]
    AVERAGE_COLS = [c.COST]
    data = pd.read_csv(PROCESSED_DATA_PATH / 'processed_sales.csv')
    data = data[[c.DATE, c.SKU, c.DESCRIPTION, c.SKU_PLATFORM, c.PLATFORM, c.SUPPLIER, c.LICENSE, c.UNITS, c.SALES_MXN, c.COST,  c.PRODUCT, c.CLUSTER]]
    data = aggregate_data(data, BOTTOM_VAR, AGGREGATION_LEVELS, VALUE_COLS, CATEGORICAL_COLS, AVERAGE_COLS)

x=1


