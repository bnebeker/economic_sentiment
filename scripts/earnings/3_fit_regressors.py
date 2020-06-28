import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import joblib
from scripts.functions import state_level_pred, mean_absolute_percentage_error, model_eval

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

df = pd.read_csv(
    './data/prepared/earnings_national.csv.tar.bz2',
    compression='bz2'
)

df_state = pd.read_csv(
    './data/prepared/earnings_by_state.csv.tar.bz2',
    compression='bz2'
)

features = df.columns[2:]
df.loc[:, 'ds'] = pd.to_datetime(df.date)

target = 'weeklynatlearn'
target_state = 'weeklyearn'

df = df[df.loc[:, target] != 0].copy()

x = df.loc[:, features]
y = df.loc[:, target]

# initialize tree & linear model
tree = DecisionTreeRegressor(max_depth=5)
lr = LinearRegression()
lasso = LassoCV(cv=5)
elastic = ElasticNetCV()


#######################################################################################################################
# #      FIT DECISION TREE
#######################################################################################################################

print('FITTING DECISION TREE...')
tree_mdl = tree.fit(x, y)
tree_preds = tree_mdl.predict(x)

tree_r2, tree_rmse, tree_mape = model_eval(y, tree_preds)

feature_imp = dict(zip(x.columns, tree_mdl.feature_importances_))
feature_imp_df = pd.DataFrame(list(feature_imp.items()), columns=['feature', 'importance'])

feature_imp_df = feature_imp_df[feature_imp_df.importance > 0].sort_values(by='importance', ascending=False)

# for feature in feature_imp_df.feature:
#     fig = plt.figure()
#     corr, _ = pearsonr(df.loc[:, feature], df.loc[:, target])
#     # ax1 = fig
#     plt.scatter(df.loc[:, feature], df.loc[:, target])
#     plt.xlabel(feature)
#     plt.ylabel(target)
#     plt.title("CORRELATION: {}".format(round(corr, 2)))
#     plt.savefig('./assets/outputs/charts/correlation_{}'.format(feature))

# plot_tree(tree_mdl, feature_names=features, fontsize=12)

df.loc[:, 'tree_prediction_{}'.format(target)] = tree_preds
df.loc[:, 'tree_preds_error'] = df.loc[:, target] - df.loc[:, 'tree_prediction_{}'.format(target)]

fig = plt.figure()
plt.hist(df.tree_preds_error)
plt.savefig('./assets/outputs/charts/earnings_tree_errors')

# FITTING DECISION TREE...
# MODEL R^2
# 0.9095699333617757
# MODEL RMSE
# 49.72811246966992

print(df.tree_preds_error.describe())
joblib.dump(tree_mdl, './assets/models/decision_tree.ml')

# apply to state level data
df_state_tree, tree_pred_name = state_level_pred(
    state_df=df_state,
    model=tree_mdl,
    features=features,
    target=target
)

## state level eval
linear_preds_state = df_state_tree.loc[:, tree_pred_name]
linear_r2, linear_rmse, linear_mape = model_eval(df_state_tree.loc[:, target_state], df_state_tree.loc[:, tree_pred_name])


# test on just top 5 states by pop
top_states = ['CA', 'TX', 'FL', 'NY', 'IL']
df_state_lim = df_state[df_state.geo.isin(top_states)]

df_state_tree, tree_pred_name = state_level_pred(
    state_df=df_state_lim,
    model=tree_mdl,
    features=features,
    target=target
)

## state level eval
linear_preds_state = df_state_tree.loc[:, tree_pred_name]
linear_r2, linear_rmse, linear_mape = model_eval(df_state_tree.loc[:, target_state], df_state_tree.loc[:, tree_pred_name])


#######################################################################################################################
# #      FIT LINEAR MODELS
#######################################################################################################################

print('FITTING LINEAR MODEL...')
linear_mdl = lr.fit(x, y)
linear_preds = linear_mdl.predict(x)

linear_r2, linear_rmse, linear_mape = model_eval(y, linear_preds)
print(cross_val_score(linear_mdl, x, y, cv=5, scoring='r2'))

df.loc[:, 'linear_prediction_{}'.format(target)] = linear_preds
df.loc[:, 'linear_preds_error'] = df.loc[:, target] - df.loc[:, 'linear_prediction_{}'.format(target)]

fig = plt.figure()
plt.hist(df.linear_preds_error)
plt.savefig('./assets/outputs/charts/earnings_lm_errors')
print(df.linear_preds_error.describe())

df_state_lm, pred_name = state_level_pred(
    state_df=df_state,
    model=linear_mdl,
    features=features,
    target=target
)

## state level eval
linear_preds_state = df_state_lm.loc[:, pred_name]
linear_r2, linear_rmse, linear_mape = model_eval(df_state_lm.loc[:, target_state], df_state_lm.loc[:, pred_name])
