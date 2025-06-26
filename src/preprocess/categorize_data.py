import sys
import unicodedata
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from src.preprocess.load_sales_data import load_sales_data
from src.utils.varnames import CATEGORIES, ColNames as c, CATEGORIES_DICT


def to_ascii(text):
    return unicodedata.normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode('ascii')

def classify_text(text):
    # Clean and extract words longer than 3 characters
    text_ascii = to_ascii(text)
    # Check if any word matches a label
    label: str | Any
    for label in CATEGORIES:
        if label in text_ascii:
            return label  # return the first matching label
    return "unknown"

def get_price_dummies(price_col, price_cell_width, contiguous_cells_num, weights_distribution):
    price_min = int(
        price_col.min() // price_cell_width) * price_cell_width - price_cell_width * contiguous_cells_num
    price_max = int(price_col.max() // price_cell_width) * price_cell_width + price_cell_width * (
                contiguous_cells_num + 1)
    price_cells_num = (price_max - price_min) // price_cell_width
    price_vector = np.array([
        assign_price_weights(price, price_min, price_cell_width, price_cells_num,
                             contiguous_cells_num, weights_distribution) for price in price_col])
    normalized_price_vector = l2_normalize(price_vector)
    colnames = [(i, i + price_cell_width) for i in range(price_min, price_max, price_cell_width)]
    return pd.DataFrame(normalized_price_vector, columns=colnames, index=price_col)

# Function to assign weights based on price and ranges
def assign_price_weights(price, price_min, price_cell_width, price_cells_num, contiguous_cells_num, weights_distribution):
    # Calculate the index of the range that the price falls into
    range_index = int((price - price_min) // price_cell_width)  # Integer division to determine the range index
    weights = np.zeros(price_cells_num)

    # Assign weights based on the range index
    # Exact range -> assign 1
    for i, v in zip(range(0, contiguous_cells_num + 1), weights_distribution):
        weights[range_index + i] = v
        weights[range_index - i] = v
    return weights

def l2_normalize(matrix):
    # Compute the L2 norm (Euclidean norm) of each row
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Normalize each row by dividing by its L2 norm
    normalized_matrix = matrix / (norms + 1e-9)  # Adding small epsilon to avoid division by zero
    return normalized_matrix

def group_data_by_sku(df):
    grouped = df.groupby([c.SKU, c.DESCRIPTION, c.SUPPLIER, c.LICENSE]).agg({c.SALES_MXN: 'sum',
                                                                                c.UNITS: 'sum'}).reset_index()
    non_unique_skus = grouped[grouped[c.SKU].duplicated(keep=False)]
    if len(non_unique_skus) > 0:
        print(non_unique_skus)
        sys.exit('Check multiple values per SKU')
    grouped[c.PRICE] = grouped[c.SALES_MXN] / grouped[c.UNITS]
    return grouped

def get_product(df):
    product = df[c.DESCRIPTION].apply(classify_text)
    return product.replace(CATEGORIES_DICT)
    # data_sku = data_sku[data_sku['product'] != 'unknown'].reset_index(drop=True)

def assign_clusters(features, cluster_num):
    kmeans = MiniBatchKMeans(n_clusters=cluster_num, random_state=0, batch_size=1024)
    return kmeans.fit_predict(features)

def get_vectors_for_clustering(df, contiguous_cells_num, weights_distribution):
    product_dummies = pd.get_dummies(df[c.PRODUCT]).astype(int)
    supplier_dummies = pd.get_dummies(df[c.SUPPLIER]).astype(int)
    license_dummies = pd.get_dummies(df[c.LICENSE]).astype(int)
    price_cell_range = 50 # TODO autocompute using dimension of other vectors
    price_dummies = get_price_dummies(df[c.PRICE], price_cell_range, contiguous_cells_num, weights_distribution)
    # TODO introduce PCA to avoid one vector from dominating the distance metric
    return np.column_stack([
        product_dummies.values.astype(int),
        supplier_dummies.values.astype(int),
        license_dummies.values.astype(int),
        price_dummies.values,
    ])

def split_clusters_by_variable(df, cluster_col, variable):
    split_cluster = pd.Series(index=df.index, dtype=int)
    new_cluster_id = 0

    for cluster_id, cluster_df in df.groupby(cluster_col):
        for _, subgroup in cluster_df.groupby(variable):
            split_cluster.loc[subgroup.index] = new_cluster_id
            new_cluster_id += 1
    return split_cluster



def categorize_data(df, contiguous_cells_num, weights_distribution, cluster_num):

    # Remove temporal dimension
    sku_df = group_data_by_sku(df)
    sku_df[c.PRODUCT] = get_product(sku_df)

    stacked_dummies = get_vectors_for_clustering(sku_df, contiguous_cells_num, weights_distribution)

    sku_df[c.CLUSTER] = assign_clusters(stacked_dummies, cluster_num)

    sku_df[c.CLUSTER] = split_clusters_by_variable(sku_df, cluster_col=c.CLUSTER, variable=[c.LICENSE, c.PRODUCT])

    return df.merge(sku_df[[c.SKU, c.PRODUCT, c.CLUSTER]], how='left', on=c.SKU)


if __name__ == '__main__':

    data_path = '/home/jmarcosh/Downloads/Fan Army (Abril).xlsx'
    supplier_exclude = ['PROVEEDOR DE PLAYERA']
    platform_include = ['Amazon', 'Mercado Libre']
    data = load_sales_data(data_path, platform_include, supplier_exclude)
    price_dummy_contiguous_cells_num = 2
    price_dummy_weights_distribution = [1, 0.5, 0.25]
    number_of_clusters = 200
    data = categorize_data(data, price_dummy_contiguous_cells_num, price_dummy_weights_distribution, number_of_clusters)