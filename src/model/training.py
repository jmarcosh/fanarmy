#%%
import re
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor, Pool
import numpy as np
from sklearn.feature_selection import RFE


from src.utils.varnames import ColNames as c
from config.config import (
    PROCESSED_DATA_PATH)


#%%
df = pd.read_csv(PROCESSED_DATA_PATH / 'data_features.csv')

#%%
features_drop = [c.DATE, c.DESCRIPTION, c.UNITS, c.SALES_MXN, c.YEAR, c.SKU_PLATFORM, 'ts_index']
features = [x for x in df.columns if x not in features_drop]
#%%

target = c.UNITS # target
categorical_features = [c.SKU, c.SKU_PLATFORM, c.SUPPLIER, c.LICENSE, c.PLATFORM, c.PRODUCT, c.CLUSTER, 'aggregation_level', 'ts_index', 'title_tatus', 'title_type', 'production_company']

df[c.DATE] = pd.to_datetime(df[c.DATE], format="%Y-%m-%d")
df[c.CLUSTER] = pd.Categorical(df[c.CLUSTER])

for col in categorical_features:
    df[col] = df[col].astype(str)
#%%
# Extract lags
lags = sorted({int(re.search(r'lag_(\d+)$', col).group(1))
               for col in features if re.match(r'lag_\d+$', col)})

# Extract trends (pairs of lags from pattern "lag_x_over_lag_y")
trends = [[int(x), int(y)]
          for col in features
          if (match := re.match(r'lag_(\d+)_over_lag_(\d+)', col))
          for x, y in [match.groups()]]

# Extract roll windows from "sales_roll_n"
rolls = sorted({int(match.group(1))
                for col in features
                if (match := re.match(r'sales_roll_(\d+)', col))})
#%%
test_date = "2025-01-01"
data = df[df[c.DATE] < test_date]
test = df[df[c.DATE] >= test_date]
data = data.sort_values(by=[c.DATE, c.SKU_PLATFORM])
#%%
from collections import Counter

def most_frequent(lst):
    counts = Counter(lst)
    max_count = max(counts.values())
    for item in lst:
        if counts[item] == max_count:
            return item
#%%
n_features_to_select = 15

# Step 1: Create time_index
data["time_index"] = data["date"].rank(method="dense").astype(int)
unique_time_steps = data["time_index"].unique()

# Step 2: Define hyperparameter grid
params = {
    'depth': 8, # , 8
    'learning_rate': 0.05, # , 0.1
    'l2_leaf_reg': 3, #, 10
    'iterations': 50,
    'early_stopping_rounds': 20
}

# Step 3: Store results
rmse_scores = []
# Step 4: Loop over hyperparameter combinations

train_idx = np.arange(9)
full_range = np.arange(0, 12)
val_idx = np.setdiff1d(full_range, train_idx)
train_time = unique_time_steps[train_idx]
grouped_sales = data[data["time_index"].isin(train_time)].groupby('ts_index')[[c.SALES_MXN, c.UNITS]].sum()
price_per_sku = (grouped_sales[c.SALES_MXN] / grouped_sales[c.UNITS]).rename(c.PRICE) # Fixed price over time. We won't have the exact price for inference
print('train periods', len(train_time))
#%%
train_nodes = []
nodes = [['bottom'], [c.CLUSTER], [c.LICENSE]]

for node in [['bottom'], [c.CLUSTER], [c.LICENSE]]:
    train = data[(data["time_index"].isin(train_time)) & (data["aggregation_level"].isin(node))].copy()
    train[c.PRICE] = train['ts_index'].map(price_per_sku)
    singleton_agg = train.groupby(['ts_index', 'aggregation_level'])['count'].mean().reset_index()
    singleton_agg_index = singleton_agg.loc[(singleton_agg['aggregation_level'] != 'bottom') & (singleton_agg['count'] == 1), 'ts_index'].tolist()
    train = train[~(train['ts_index'].isin(singleton_agg_index))]
    train_nodes.append(train)
#%%
# Feature Importance
val_nodes = []
for node, train in zip(nodes, train_nodes):

    categorical_features = [x for x in categorical_features if x in features]
    train_pool = Pool(train[features], label=train[target], cat_features=categorical_features)
    train_val = train
    preds = []
    val_dfs = []
    for i in val_idx:
        val_time = unique_time_steps[i]
        train_val = pd.concat([train_val, data[(data["time_index"] == val_time) & (data["aggregation_level"].isin(node))]])
        train_val[c.PRICE] = train_val['ts_index'].map(price_per_sku)

        for lag in lags:
            train_val[f'lag_{lag}'] = train_val.groupby(['ts_index', 'cont_sales_id'], observed=True)[c.UNITS].shift(lag)
        for trend in trends:
            train_val[f'lag_{trend[0]}_over_lag_{trend[1]}'] = (train_val[f'lag_{trend[0]}'] / train_val[f'lag_{trend[1]}']).replace([float('inf'), -float('inf')], np.nan)
        for roll in rolls:
            train_val[f'sales_roll_{roll}'] = train_val.groupby(['ts_index', 'cont_sales_id'], observed=True)[c.UNITS].transform(
        lambda x: x.shift(1).rolling(window=roll, min_periods=1).mean())

        val_i = train_val[train_val["time_index"] == val_time].copy()
        val_pool_i = Pool(val_i[features], label=val_i[target], cat_features=categorical_features)
        model = CatBoostRegressor(**params, verbose=0)

        model.fit(train_pool, eval_set=val_pool_i)
        feature_importances = model.feature_importances_

        preds_i = model.predict(val_i[features])
        train_val.loc[train_val["time_index"]== val_time, c.UNITS] = preds_i
        preds += preds_i.tolist()
        val_dfs.append(val_i)
        # rmse = mean_squared_error(val_i[target], preds_i, squared=False)
        # rmse_scores.append(rmse)

    val = pd.concat(val_dfs)
    val['predictions'] = preds
    val['errors'] = val[c.UNITS] - preds
    val_nodes.append(val)

#%%
one_model = pd.concat(val_nodes).copy()
#%%
three_models = pd.concat(val_nodes).copy()

#%%
data[data[c.SKU_PLATFORM] == 'PP9652LS_Amazon']
#%%
rmse_scores = np.sqrt(mean_squared_error(val[target], preds))

#%%
for i, df in enumerate(val_nodes):
    if df.isin([float('inf'), -float('inf')]).any().any():
        print(f"DataFrame {i} contains inf values.")
#%%
np.sqrt(mean_squared_error(one_model[target], one_model['predictions']))
#%%
np.sqrt(mean_squared_error(three_models[target], three_models['predictions']))

#%%
val['errors'] = val[c.UNITS] - preds
#%%
# Recursive feature Elimination


# model_eval = CatBoostRegressor(**params, verbose=0)
# model.fit(train_pool) # , eval_set=val_pool_i  # Why results improve when introducing to the loop

removed_seq = []
features_lst = []
feature_importance_lst = []
while len(features) > n_features_to_select:
    categorical_features = [x for x in categorical_features if x in features]
    train_pool = Pool(train[features], label=train[target], cat_features=categorical_features)
    train_val = train
    preds = []
    val_dfs = []
    fold_remove = []
    for i in val_idx:
        val_time = unique_time_steps[i]
        train_val = pd.concat([train_val, data[data["time_index"] == val_time]])
        train_val[c.PRICE] = train_val[c.SKU_PLATFORM].map(price_per_sku)

        for lag in lags:
            train_val[f'lag_{lag}'] = train_val.groupby([c.SKU_PLATFORM, 'cont_sales_id'], observed=True)[c.UNITS].shift(lag)
        for trend in trends:
            train_val[f'lag_{trend[0]}_over_lag_{trend[1]}'] = train_val[f'lag_{trend[0]}'] / train_val[f'lag_{trend[1]}']
        for roll in rolls:
            train_val[f'sales_roll_{roll}'] = train_val.groupby([c.SKU_PLATFORM, 'cont_sales_id'], observed=True)[c.UNITS].transform(
        lambda x: x.shift(1).rolling(window=roll, min_periods=1).mean())

        val_i = train_val[train_val["time_index"] == val_time].copy()
        val_pool_i = Pool(val_i[features], label=val_i[target], cat_features=categorical_features)
        model = CatBoostRegressor(**params, verbose=0)

        model.fit(train_pool, eval_set=val_pool_i)
        feature_importances = model.feature_importances_
        fold_remove.append(np.argmin(feature_importances))

        preds_i = model.predict(val_i[features])
        train_val.loc[train_val["time_index"]== val_time, c.UNITS] = preds_i
        preds += preds_i.tolist()
        val_dfs.append(val_i)
        # rmse = mean_squared_error(val_i[target], preds_i, squared=False)
        # rmse_scores.append(rmse)

    val = pd.concat(val_dfs)
    rmse_scores.append(np.sqrt(mean_squared_error(val[target], preds)))
    features_lst.append(features.copy())
    feature_importance_lst.append(feature_importances)
    removed_seq.append(features.pop(most_frequent(fold_remove)))
    print()


#%%
import pandas as pd

# Your input



# Step 1: Get all unique categories
all_categories = sorted(set(cat for sublist in features_lst for cat in sublist))

# Step 2: Build a dictionary for each observation
columns = []
for cats, vals in zip(features_lst, feature_importance_lst):
    col_data = dict(zip(cats, vals))
    full_col = {cat: col_data.get(cat, None) for cat in all_categories}
    columns.append(full_col)

# Step 3: Create DataFrame
fi = pd.DataFrame(columns).T
fi.columns = [f'obs_{i}' for i in range(len(columns))]  # Optional: name columns


#%%
fi.to_csv('/home/jmarcosh/Downloads/fi.csv')
#%%
