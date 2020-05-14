import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

df = pd.read_csv(
    './data/prepared_data_full_us.csv.tar.bz2',
    compression='bz2'
)

features = df.columns[3:]
df.loc[:, 'ds'] = pd.to_datetime(df.date)

target = 'target_bus12'
# target = 'target_umex'

x = df.loc[:, features]
y = df.loc[:, target]

# initialize tree & linear model
tree = DecisionTreeRegressor(max_depth=5)
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

    print("MODEL RMSE")
    print(model_rmse)

    return model_r2, model_rmse, model_mape


print('FITTING DECISION TREE...')
tree_mdl = tree.fit(x, y)
tree_preds = tree_mdl.predict(x)

tree_r2, tree_rmse, tree_mape = model_eval(y, tree_preds)

feature_imp = dict(zip(x.columns, tree_mdl.feature_importances_))
feature_imp_df = pd.DataFrame(list(feature_imp.items()), columns=['feature', 'importance'])

feature_imp_df = feature_imp_df[feature_imp_df.importance > 0].sort_values(by='importance', ascending=False)

for feature in feature_imp_df.feature:
    fig = plt.figure()
    corr, _ = pearsonr(df.loc[:, feature], df.loc[:, target])
    # ax1 = fig
    plt.scatter(df.loc[:, feature], df.loc[:, target])
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.title("CORRELATION: {}".format(round(corr, 2)))
    plt.savefig('./assets/outputs/charts/correlation_{}'.format(feature))

# plot_tree(tree_mdl, feature_names=features, fontsize=12)

df.loc[:, 'tree_prediction_{}'.format(target)] = tree_preds
df.loc[:, 'tree_preds_error'] = df.loc[:, 'target_bus12'] - df.loc[:, 'tree_prediction_target_bus12']

fig = plt.figure()
plt.hist(df.tree_preds_error)
plt.savefig('./assets/outputs/charts/tree_errors')

# FITTING DECISION TREE...
# MODEL R^2
# 0.9095699333617757
# MODEL RMSE
# 49.72811246966992

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
    'unemployment_insurance_lag1',
    'unemployment_insurance',
    'unemployment_insurance_lag2',
    'wage_growth',
    'wage_growth_lag2',
    'unemployment_insurance',
    'unemployment_lag1'
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

ax = df.plot(x="ds", y=target, legend=False)
ax2 = ax.twinx()
df.plot(x="ds", y="linear_prediction_target_bus12", ax=ax2, legend=False, color="r")
ax.figure.legend()
plt.show()


