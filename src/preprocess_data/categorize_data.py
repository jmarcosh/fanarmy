import unicodedata
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from src.preprocess_data.utils.sales_data_functions import load_sales_data
from src.preprocess_data.utils.varnames import CATEGORIES, ColNames, CATEGORIES_DICT


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

def get_price_dummies(price_col, price_cell_width, neighbors_number, neighbor_weights):
    price_min = int(
        price_col.min() // price_cell_width) * price_cell_width - price_cell_width * contiguous_cells_num
    price_max = int(price_col.max() // price_cell_width) * price_cell_width + price_cell_width * (
                contiguous_cells_num + 1)
    price_cells_num = (price_max - price_min) // price_cell_width
    price_vector = np.array([assign_price_weights(price, price_min, price_cell_width, price_cells_num, neighbors_number, neighbor_weights) for price in price_col])
    normalized_price_vector = l2_normalize(price_vector)
    colnames = [(i, i + price_cell_width) for i in range(price_min, price_max, price_cell_width)]
    return pd.DataFrame(normalized_price_vector, columns=colnames, index=price_col.index)

# Function to assign weights based on price and ranges
def assign_price_weights(price, price_min, price_cell_width, price_cells_num, neighbors_number, neighbor_weights):
    # Calculate the index of the range that the price falls into
    range_index = int((price - price_min) // price_cell_width)  # Integer division to determine the range index
    weights = np.zeros(price_cells_num)

    # Assign weights based on the range index
    # Exact range -> assign 1
    for i, v in zip(range(0, neighbors_number + 1), neighbor_weights):
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
    grouped = df.groupby([c.SKU, c.DESCRIPTION, c.SUPPLIER, c.LICENSE]).agg({c.SALES_MXN: 'sum', c.COST: 'mean',
                                                                                c.UNITS: 'sum'}).reset_index()
    grouped[c.PRICE] = grouped[c.SALES_MXN] / grouped[c.UNITS]
    return grouped

def get_product(df):
    product = df[c.DESCRIPTION].apply(classify_text)
    return product.replace(CATEGORIES_DICT)
    # data_sku = data_sku[data_sku['product'] != 'unknown'].reset_index(drop=True)

def assign_clusters(features, cluster_num):
    kmeans = MiniBatchKMeans(n_clusters=cluster_num, random_state=0, batch_size=1024)
    return kmeans.fit_predict(features)

def get_vectors_for_clustering(df, price_cell_range, contiguous_cells_num, weights_distribution):
    product_dummies = pd.get_dummies(df[c.PRODUCT]).astype(int)
    supplier_dummies = pd.get_dummies(df[c.SUPPLIER]).astype(int)
    license_dummies = pd.get_dummies(df[c.LICENSE]).astype(int)
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




data_path = '/home/jmarcosh/Downloads/Fan Army (Abril).xlsx'
supplier_exclude = ['PROVEEDOR DE PLAYERA']
platform_include = ['Amazon', 'Mercado Libre']
c = ColNames()

data = load_sales_data(data_path, platform_include, supplier_exclude)
sku_data = group_data_by_sku(data)
sku_data[c.PRODUCT] = get_product(sku_data)


stacked_dummies = get_vectors_for_clustering(sku_data, price_cell_range, contiguous_cells_num, weights_distribution)

sku_data[c.CLUSTER] = assign_clusters(stacked_dummies, cluster_num)

sku_data[c.CLUSTER] = split_clusters_by_variable(sku_data, cluster_col=c.CLUSTER, variable=[c.LICENSE, c.PRODUCT])

data = data.merge(sku_data[[c.SKU, c.PRODUCT, c.CLUSTER]], how='left', on=c.SKU)


price_cell_range = 50
contiguous_cells_num = 2
weights_distribution = [1, 0.5, 0.25]
cluster_num = 200