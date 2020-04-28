import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

# initialize tree & linear model
tree = DecisionTreeRegressor(max_depth=3)
lr = LinearRegression()


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


df = pd.read_csv(
    './data/prepared_data_full_us.csv.tar.bz2',
    compression='bz2'
)

target = 'target_bus12'
# target = 'target_umex'

features = df.columns[3:]

x = df.loc[:, features]
y = df.loc[:, target]

print('FITTING DECISION TREE...')
tree_mdl = tree.fit(x, y)
tree_preds = tree_mdl.predict(x)

tree_r2 = r2_score(y, tree_preds)
tree_rmse = mean_squared_error(y, tree_preds)
tree_mape = mean_absolute_percentage_error(y, tree_preds)

print("TREE R^2:")
print(tree_r2)

print('FITTING LINEAR MODEL...')
linear_mdl = lr.fit(x, y)
linear_preds = linear_mdl.predict(x)

linear_r2 = r2_score(y, linear_preds)
linear_rmse = mean_squared_error(y, linear_preds)
linear_mape = mean_absolute_percentage_error(y, linear_preds)
