import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
import numpy as np

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

df = pd.read_csv(
    './data/prepared_data_full_us.csv.tar.bz2',
    compression='bz2'
)

target = 'target_bus12'
# target = 'target_umex'

features = df.columns[3:]

x = df.loc[:, features]
y = df.loc[:, target]

# initialize tree & linear model
tree = DecisionTreeRegressor(max_depth=3)
lr = LinearRegression()
lasso = LassoCV(cv=5)
elastic = ElasticNetCV()


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def model_eval(y_true=y, y_pred=None):
    model_r2 = r2_score(y_true, y_pred)
    model_rmse = mean_squared_error(y_true, y_pred)
    model_mape = mean_absolute_percentage_error(y_true, y_pred)

    print("MODEL R^2")
    print(model_r2)

    return model_r2, model_rmse, model_mape

target = 'target_bus12'
# target = 'target_umex'

features = df.columns[3:]

x = df.loc[:, features]
y = df.loc[:, target]

print('FITTING DECISION TREE...')
tree_mdl = tree.fit(x, y)
tree_preds = tree_mdl.predict(x)

tree_r2, tree_rmse, tree_mape = model_eval(y, tree_preds)

tree_mdl.feature_importances_
# plot_tree(tree_mdl, feature_names=features, fontsize=12)

df.loc[:, 'tree_prediction_target_bus12'] = tree_preds
df.loc[:, 'tree_preds_error'] = df.loc[:, 'target_bus12'] - df.loc[:, 'tree_prediction_target_bus12']

# FITTING DECISION TREE...
# MODEL R^2
# 0.6633697461005208

print('FITTING LINEAR MODEL...')
linear_mdl = lr.fit(x, y)
linear_preds = linear_mdl.predict(x)

linear_r2, linear_rmse, linear_mape = model_eval(y, linear_preds)
print(cross_val_score(linear_mdl, x, y, cv=5, scoring='r2'))

df.loc[:, 'linear_prediction_target_bus12'] = linear_preds
df.loc[:, 'linear_preds_error'] = df.loc[:, 'target_bus12'] - df.loc[:, 'linear_prediction_target_bus12']

print('FITTING LASSO LINEAR MODEL...')
lasso_mdl = lasso.fit(x, y)
lasso_preds = lasso_mdl.predict(x)

lasso_r2, lasso_rmse, lasso_mape = model_eval(y, lasso_preds)

print("ELASTICNET MODEL...")
en_mdl = elastic.fit(x, y)
en_preds = en_mdl.predict(x)

en_r2, en_rmse, en_mape = model_eval(y, en_preds)


# linear model on subset of features
limit_features = [
    'stock_market',
    'jobs',
    'unemployment',
    'stocks',
    'unemployment_trend'
]

x = df.loc[:, limit_features]
print('FITTING LINEAR MODEL...')
linear_mdl = lr.fit(x, y)
linear_preds = linear_mdl.predict(x)

linear_r2, linear_rmse, linear_mape = model_eval(y, linear_preds)
print(cross_val_score(linear_mdl, x, y, cv=5))

df.to_csv(
    './assets/outputs/df_with_preds.csv.tar.bz2',
    compression='bz2',
    index=False
)

