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

features = df.columns[3:]

output = pd.DataFrame(columns=[
    'feature',
    'target',
    'cardinality',
    'mean',
    'stddev',
    'min',
    'p5',
    'p25',
    'p50',
    'p75',
    'p95',
    'max',
    'kurtosis',
    'skew',
    'tree_r2',
    'tree_rmse',
    'tree_mape',
    'linear_r2',
    'linear_rmse',
    'linear_mape'
])

target_list = ['target_bus12', 'target_umex']

for target in target_list:
    for f in features:
        print(f)
        x = df.loc[:, f]
        x = np.array(x).reshape(-1, 1)
        y = df.loc[:, target]

        print('FITTING DECISION TREE...')
        tree_mdl = tree.fit(x, y)
        tree_preds = tree_mdl.predict(x)

        tree_r2 = r2_score(y, tree_preds)
        tree_rmse = mean_squared_error(y, tree_preds)
        tree_mape = mean_absolute_percentage_error(y, tree_preds)

        print('FITTING LINEAR MODEL...')
        linear_mdl = lr.fit(x, y)
        linear_preds = linear_mdl.predict(x)

        linear_r2 = r2_score(y, linear_preds)
        linear_rmse = mean_squared_error(y, linear_preds)
        linear_mape = mean_absolute_percentage_error(y, linear_preds)

        # general feature metrics
        cardinality = df.loc[:, f].nunique()
        col_mean, stddev, min, p25, p50, p75, max = df.loc[:, f].describe()[[1, 2, 3, 4, 5, 6, 7]]
        skew = df.loc[:, f].skew()
        kurt = df.loc[:, f].kurt()
        p5 = df.loc[:, f].quantile(0.05)
        p95 = df.loc[:, f].quantile(0.95)

        _tmp = pd.DataFrame(columns=[
            'feature',
            'target',
            'cardinality',
            'mean',
            'stddev',
            'min',
            'p5',
            'p25',
            'p50',
            'p75',
            'p95',
            'max',
            'kurtosis',
            'skew',
            'tree_r2',
            'tree_rmse',
            'tree_mape',
            'linear_r2',
            'linear_rmse',
            'linear_mape'
        ])

        _tmp.loc[0] = [
            f, target, cardinality, col_mean, stddev, min, p5, p25, p50,
            p75, p95, max, kurt, skew, tree_r2, tree_rmse, tree_mape,
            linear_r2, linear_rmse, linear_mape
        ]

        output = output.append(_tmp, ignore_index=True)

output.sort_values(by='linear_r2', ascending=False, inplace=True)
print(output.head(10))

output.to_csv(
    './assets/outputs/single_feature_report.csv.tar.bz2'.format(target),
    compression='bz2',
    index=False
)
