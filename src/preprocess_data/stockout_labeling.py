from src.preprocess_data.utils.sales_data_functions import load_sales_data
from src.preprocess_data.utils.varnames import ColNames

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from sklearn.metrics import mean_squared_error




def compute_rule_based_stockout_indicator(df, surge_value, dev_0, x, dev_x):
    df = df.copy()
    df['surge'] = (df['residuals'] > surge_value).astype(int)
    is_first = df.groupby([c.SKU_PLATFORM], observed=True).cumcount() == 0
    surge_t_minus_1 = df.groupby([c.SKU_PLATFORM], observed=True)['surge'].shift(1).fillna(0)
    surge_t_plus_1 = df.groupby([c.SKU_PLATFORM], observed=True)['surge'].shift(-1).fillna(0)
    a = - (dev_x - dev_0) / np.log(1 + x)
    slump = (df['residuals'] < a * np.log(1 + df[c.UNITS]) - dev_0).astype(int)
    df['stockout'] = slump & ((surge_t_minus_1 == 1) | (surge_t_plus_1 == 1) | is_first).astype(int)
    print('Initial stockout', df['stockout'].sum())

    date_max = df[c.DATE].max()
    date_min = df[c.DATE].min()
    max_range = (date_max.year - date_min.year) * 12 + (date_max.month - date_min.month)
    count = 0

    while count < max_range:
        stockout_t_minus_1 = df.groupby(['sku_platform'], observed=True)['stockout'].shift(1).fillna(0)
        stockout_t_plus_1 = df.groupby(['sku_platform'], observed=True)['stockout'].shift(-1).fillna(0)
        condition = ((stockout_t_minus_1 == 1) | (stockout_t_plus_1 == 1)) & slump
        df.loc[condition, 'stockout'] = 1
        print(count, df['stockout'].sum())
        count += 1

    # Update last stockout to keep 1s that are:
    # # - part of a contiguous block (have a neighbor 1),
    # # - OR are the first observation
    stockout_t_minus_1 = df.groupby(['sku_platform'], observed=True)['stockout'].shift(1).fillna(0)
    stockout_t_plus_1 = df.groupby(['sku_platform'], observed=True)['stockout'].shift(-1).fillna(0)
    keep_mask = ((((stockout_t_minus_1 == 1) | (stockout_t_plus_1 == 1)) | is_first)
    )
    # Set other 1s to 0
    stockout = df['stockout'].where(keep_mask, 0)
    print(count + 1, stockout.sum())
    return stockout


def filter_out_skus_with_non_significant_sales(df, sales_threshold):
    data_sku_platform = df.groupby([
        c.SKU_PLATFORM], observed=True)[[c.UNITS]].sum().reset_index()
    sorted_sales = data_sku_platform.sort_values(by=[c.UNITS], ascending=False)
    sorted_sales_array = sorted_sales[c.UNITS].values
    # % of SKUs
    sku_pct = np.arange(1, len(sorted_sales_array) + 1) / len(sorted_sales_array)
    # Cumulative sales as % of total
    cum_sales_pct = np.cumsum(sorted_sales_array) / sorted_sales_array.sum()
    # Find index of first value greater than the threshold
    index = np.argmax(sorted_sales_array == sales_threshold)
    selected_skus = sorted_sales[c.SKU_PLATFORM].iloc[:index].tolist()
    dfs = df[df[c.SKU_PLATFORM].isin(selected_skus)].reset_index(drop=True)
    print(f'Percentage of SKUs with sales above {sales_threshold}: {sku_pct[index]:.2%}')
    print(f'Percentage of rows kept: {len(dfs) / len(df)}')
    print(f'Percentage of sales in SKUs above threshold: {cum_sales_pct[index]:.2%}')
    return dfs



def fit_model_and_predict(df):
    poisson_model = smf.glm(
        formula=f'{c.UNITS} ~ {c.PRICE} + {c.PLATFORM_MONTHLY_SALES} + C({c.SKU_PLATFORM})',
        data=df,
        family=sm.families.Poisson()
    ).fit()
    y_pred = poisson_model.predict(df)
    rmse = np.sqrt(mean_squared_error(df[c.UNITS], y_pred))
    print('Poisson model RMSE:', round(rmse, 2))
    return y_pred


def add_features_for_stockout_fixed_effects_model(df):
    df[c.PRICE] = df[c.SALES_MXN] / df[c.UNITS]
    df[c.PRICE] = df[c.PRICE].fillna(df.groupby(c.SKU)[c.PRICE].transform('mean'))
    # Add platform monthly sales as another feature
    platform_monthly_sales = (df.groupby([c.DATE, c.PLATFORM], observed=True)[[c.UNITS]].sum().
                              rename(columns={c.UNITS: c.PLATFORM_MONTHLY_SALES}).reset_index())
    df = df.merge(platform_monthly_sales, on=['date', 'Plataforma'], how='left')
    # Include sku_platform_fixed_effects
    df[c.SKU_PLATFORM] = df[c.SKU].astype(str) + "_" + df[c.PLATFORM].astype(str)
    df[c.SKU_PLATFORM] = df[c.SKU_PLATFORM].astype('category')
    return df



def stockout_labeling(df, minimum_sales_per_sku, dev_sale_zero, second_point, dev_second_point, surge_threshold):

    #Load data
    df = add_features_for_stockout_fixed_effects_model(df)

    # Filter out non significant sales
    df = filter_out_skus_with_non_significant_sales(df, minimum_sales_per_sku)

    # Fit and predict the model
    df['pred'] = fit_model_and_predict(df)

    # Pearson residuals measures how many standard deviations the prediction is from the real value
    df['residuals'] = (df[c.UNITS] - df['pred']) / np.sqrt(df['pred'])

    # Convert predictions to stockout indicator
    df['stockout'] = compute_rule_based_stockout_indicator(df, surge_threshold, dev_sale_zero, second_point, dev_second_point)


    return df




if __name__ == '__main__':
    data_path = '/home/jmarcosh/Downloads/Fan Army (Abril).xlsx'
    supplier_exclude = ['PROVEEDOR DE PLAYERA']
    platform_include = ['Amazon', 'Mercado Libre']
    dev_sale_zero = 1
    second_point = 10
    dev_second_point = 2
    surge_threshold = 1.25
    minimum_sales_per_sku = 10
    c = ColNames()
    data = load_sales_data(data_path, platform_include, supplier_exclude)
    data = stockout_labeling(data, minimum_sales_per_sku, dev_sale_zero, second_point,
                      dev_second_point, surge_threshold)

