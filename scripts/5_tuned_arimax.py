import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import itertools
import random
from pandas.plotting import autocorrelation_plot
from scripts.functions import ts_evaluation, mape
import joblib

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

df = pd.read_csv(
    './data/prepared_data_full_us.csv.tar.bz2',
    compression='bz2'
)

print(df.shape)
target = 'target_bus12'

features = df.columns[3:]

df.loc[:, 'ds'] = pd.to_datetime(df.date)
df_model = df.loc[:, ['ds', target]]

df_model.columns = ['ds', 'y']

train_size = int(df_model.shape[0] * .8)
test_size = df_model.shape[0] - train_size
train_df = df_model.head(train_size)
test_df = df_model.tail(test_size)

train_exog = df.loc[:, features].head(train_size)
test_exog = df.loc[:, features].tail(test_size)

test_min = test_df.ds.min()


# #1, MAPE ~7.1%
# BEST PARAMETERS
param = (8, 1, 6)

mod = sm.tsa.statespace.SARIMAX(
    train_df.y,
    exog=train_exog,
    order=param,
    enforce_stationarity=False,
    enforce_invertibility=False
)

arimax_model = mod.fit()
aic_tmp = arimax_model.aic
print('ARIMA{} - AIC:{}'.format(param, arimax_model.aic))

full_preds, forecast, m = ts_evaluation(model=arimax_model, train_df=train_df, test_df=test_df, test_exog=test_exog)

ax = full_preds.plot(x="ds", y='y', legend=False)
ax2 = ax.twinx()
full_preds.plot(x="ds", y="yhat", ax=ax2, legend=False, color="r")
ax.figure.legend()
plt.show()

# serialize best model
joblib.dump(arimax_model, './assets/models/arimax.ml')

# OTHER TESTS:
# #2, MAPE ~7.2%
# pretty much all errors are low yhat < y
param = (4, 2, 5)

mod = sm.tsa.statespace.SARIMAX(
    train_df.y,
    exog=train_exog,
    order=param,
    enforce_stationarity=False,
    enforce_invertibility=False
)

arimax_model = mod.fit()
aic_tmp = arimax_model.aic
print('ARIMA{} - AIC:{}'.format(param, arimax_model.aic))

full_preds, forecast, m = ts_evaluation(model=arimax_model, train_df=train_df, test_df=test_df, test_exog=test_exog)

ax = full_preds.plot(x="ds", y='y', legend=False)
ax2 = ax.twinx()
full_preds.plot(x="ds", y="yhat", ax=ax2, legend=False, color="r")
ax.figure.legend()
plt.show()
