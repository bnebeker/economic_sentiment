import pandas as pd
import numpy as np
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
import matplotlib.pyplot as plt
import statsmodels.api as sm
import itertools
import random
from pandas.plotting import autocorrelation_plot
from scripts.functions import ts_evaluation, mape

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

# df_model.plot()
# plt.show()

# ac_plot = autocorrelation_plot(df_model.y)
# plt.show()
# ac_plot.figure.savefig('./assets/outputs/charts/autocorrelation.png')

# sd = sm.tsa.seasonal_decompose(df_model.y, model='add', freq=192)
# sd_plot = sd.plot()
# plt.show()
# sd_plot.savefig('./assets/outputs/charts/seasonal_decomposition.png')

train_size = int(df_model.shape[0] * .8)
test_size = df_model.shape[0] - train_size
train_df = df_model.head(train_size)
test_df = df_model.tail(test_size)

train_exog = df.loc[:, features].head(train_size)
test_exog = df.loc[:, features].tail(test_size)

test_min = test_df.ds.min()


########################################################################################
#      ARIMAX - EXOG, NO SEASONAL
########################################################################################

# single
param = (1, 1, 1)

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

full_preds, forecast = ts_evaluation(model=arimax_model, train_df=train_df, test_df=test_df, test_exog=test_exog)

ax = full_preds.plot(x="ds", y='y', legend=False)
ax2 = ax.twinx()
full_preds.plot(x="ds", y="yhat", ax=ax2, legend=False, color="r")
ax.figure.legend()
plt.show()


# RANDOM SEARCH
p = d = q = list(range(0, 25))
pdq = list(itertools.product(p, d, q))

best_score = float("inf")
best_config = []

# param_results = pd.DataFrame(columns=[
#     'p',
#     'd',
#     'q',
#     'aic',
#     'test_mape'
# ])
# read in last run and append to it
param_results = pd.read_csv('./assets/outputs/arimax_tuning.csv')

# PROXY RANDOM SEARCH

for i in range(0, 500):
    param = random.choice(pdq)

    print("ITERATION {}".format(i))
    print(param)

    try:
        mod = sm.tsa.statespace.SARIMAX(
            train_df.y,
            exog=train_exog,
            order=param,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        results = mod.fit()
        aic_tmp = results.aic
        print('ARIMA{} - AIC:{}'.format(param, results.aic))
        if best_score > aic_tmp:
            best_score = aic_tmp

        _p = param[0]
        _d = param[1]
        _q = param[2]
        _, _, mape_output = ts_evaluation(model=results, train_df=train_df, test_df=test_df, test_exog=test_exog)
        _param_results = pd.DataFrame(columns=[
            'p',
            'd',
            'q',
            'aic',
            'test_mape'
        ])
        _param_results.loc[0] = [_p, _d, _q, aic_tmp, mape_output]

        param_results = param_results.append(_param_results, ignore_index=True)

        param_results.to_csv(
            './assets/outputs/arimax_tuning.csv',
            index=False
        )
    except:
        continue

    i += 1

print(param_results.sort_values(by='test_mape').head())

###########################################################################################
# test some top arima AIC score combos
###########################################################################################
# 716  4.0  2.0  5.0  1160.458098   7.019894
# 181  8.0  1.0  6.0  1148.145325   7.117879
# 53   5.0  2.0  5.0  1162.852995   7.245784
# 133  2.0  2.0  2.0  1185.913040   7.389793
# 102  8.0  2.0  2.0  1152.863219   7.396897
# 37   4.0  1.0  4.0  1174.818365   7.576425

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




########################################################################################
#      SARIMAX
########################################################################################

# single
param = (1, 1, 1)
param_seasonal = (1, 1, 1, 2)

mod = sm.tsa.statespace.SARIMAX(
    train_df.y,
    exog=train_exog,
    order=param,
    seasonal_order=param_seasonal,
    enforce_stationarity=False,
    enforce_invertibility=False
)

sarimax_model = mod.fit()
aic_tmp = sarimax_model.aic
print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, sarimax_model.aic))

sarimax_forecast = test_df
preds = sarimax_model.forecast(test_df.shape[0], exog=test_exog)
sarimax_forecast.loc[:, 'yhat'] = preds
sarimax_forecast.loc[:, 'y_error'] = sarimax_forecast.loc[:, 'y'] - sarimax_forecast.loc[:, 'yhat']
sarimax_forecast.loc[:, 'y_pct_error'] = sarimax_forecast.loc[:, 'y_error'] / sarimax_forecast.loc[:, 'y']
sarimax_forecast.loc[:, 'y_abs_pct_error'] = abs(sarimax_forecast.loc[:, 'y_pct_error'])
# arima_forecast.loc[:, 'y_in_yhat_band'] = np.where((arima_forecast.y >= arima_forecast.yhat_lower) & (arima_forecast.y <= arima_forecast.yhat_upper), 1, 0)

print(sarimax_forecast.y_abs_pct_error.describe())

print("MAPE:")
print(mape(sarimax_forecast.y, sarimax_forecast.yhat))
# MAPE = 11%

sarimax_train_preds = sarimax_model.predict(full_results=True)

sarimax_df_train = pd.concat([train_df, sarimax_train_preds], axis=1)
sarimax_df_train.columns = ['ds', 'y', 'yhat']

# cut after a couple periods
sarimax_df_train = sarimax_df_train[sarimax_df_train.ds >= '2004-04-01']

full_preds = sarimax_df_train.append(sarimax_forecast.loc[:, ['ds', 'y', 'yhat']])

ax = full_preds.plot(x="ds", y='y', legend=False)
ax2 = ax.twinx()
full_preds.plot(x="ds", y="yhat", ax=ax2, legend=False, color="r")
ax.figure.legend()
plt.show()


p = d = q = [0, 1, 2, 3, 4, 6, 8]
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
weekly_pdq = [(x[0], x[1], x[2], 52) for x in list(itertools.product(p, d, q))]
seasonal_pdq.extend(weekly_pdq)

best_score = float("inf")
best_config = []

param_results = pd.DataFrame(columns=[
    'p',
    'd',
    'q',
    'ps',
    'ds',
    'qs',
    's',
    'aic'
])

# PROXY RANDOM SEARCH

for i in range(0, 100):
    param = random.choice(pdq)
    param_seasonal = random.choice(seasonal_pdq)

    print("ITERATION {}".format(i))
    print(param)
    print(param_seasonal)

    try:
        mod = sm.tsa.statespace.SARIMAX(
            train_df.y,
            order=param,
            seasonal_order=param_seasonal,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        results = mod.fit()
        aic_tmp = results.aic
        print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
        if best_score > aic_tmp:
            best_score = aic_tmp
            best_config = param_seasonal

        _p = param[0]
        _d = param[1]
        _q = param[2]
        _ps = param_seasonal[0]
        _ds = param_seasonal[1]
        _qs = param_seasonal[2]
        _s = param_seasonal[3]
        _param_results = pd.DataFrame(columns=[
            'p',
            'd',
            'q',
            'ps',
            'ds',
            'qs',
            's',
            'aic'
        ])
        _param_results.loc[0] = [_p, _d, _q, _ps, _ds, _qs, _s, aic_tmp]

        param_results = param_results.append(_param_results, ignore_index=True)

        param_results.to_csv(
            './assets/outputs/sarimax_tuning.csv',
            index=False
        )
    except:
        continue

    i += 1

###########################################################################################
# test some top arima AIC score combos
###########################################################################################

# 1
# 2.0,3.0,1.0,6.0,6.0,0.0,12.0,129.7791765363384

param = (2, 3, 1)
param_seasonal = (6, 6, 0, 12)
mod = sm.tsa.statespace.SARIMAX(
    train_df.y,
    order=param,
    seasonal_order=param_seasonal,
    enforce_stationarity=False,
    enforce_invertibility=False
)

mdl1 = mod.fit()
print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, mdl1.aic))
arima_forecast = test_df
preds = mdl1.forecast(test_df.shape[0], exog=test_exog)
arima_forecast.loc[:, 'yhat'] = preds
arima_forecast.loc[:, 'y_error'] = arima_forecast.loc[:, 'y'] - arima_forecast.loc[:, 'yhat']
arima_forecast.loc[:, 'y_pct_error'] = arima_forecast.loc[:, 'y_error'] / arima_forecast.loc[:, 'y']
arima_forecast.loc[:, 'y_abs_pct_error'] = abs(arima_forecast.loc[:, 'y_pct_error'])
# arima_forecast.loc[:, 'y_in_yhat_band'] = np.where(
# (arima_forecast.y >= arima_forecast.yhat_lower) & (arima_forecast.y <= arima_forecast.yhat_upper), 1, 0)

print(arima_forecast.y_abs_pct_error.describe())

print("MAPE:")
print(mape(arima_forecast.y, arima_forecast.yhat))
# MAPE:

sarimax_train_preds = mdl1.predict(full_results=True)

sarimax_df_train = pd.concat([train_df, sarimax_train_preds], axis=1)
sarimax_df_train.columns = ['ds', 'y', 'yhat']

# cut after a couple periods
sarimax_df_train = sarimax_df_train[sarimax_df_train.ds >= '2004-04-01']

full_preds = sarimax_df_train.append(sarimax_forecast.loc[:, ['ds', 'y', 'yhat']])

ax = full_preds.plot(x="ds", y='y', legend=False)
ax2 = ax.twinx()
full_preds.plot(x="ds", y="yhat", ax=ax2, legend=False, color="r")
ax.figure.legend()
plt.show()




param = (6, 2, 8)
param_seasonal = (0, 0, 3, 52)
mod = sm.tsa.statespace.SARIMAX(
    train_df.y,
    order=param,
    seasonal_order=param_seasonal,
    enforce_stationarity=False,
    enforce_invertibility=False
)

arima_model = mod.fit()
print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, arima_model.aic))
arima_forecast = test_df
preds = arima_model.forecast(test_df.shape[0])
arima_forecast.loc[:, 'yhat'] = preds
arima_forecast.loc[:, 'y_error'] = arima_forecast.loc[:, 'y'] - arima_forecast.loc[:, 'yhat']
arima_forecast.loc[:, 'y_pct_error'] = arima_forecast.loc[:, 'y_error'] / arima_forecast.loc[:, 'y']
arima_forecast.loc[:, 'y_abs_pct_error'] = abs(arima_forecast.loc[:, 'y_pct_error'])
# arima_forecast.loc[:, 'y_in_yhat_band'] = np.where((arima_forecast.y >= arima_forecast.yhat_lower) & (arima_forecast.y <= arima_forecast.yhat_upper), 1, 0)

print(arima_forecast.y_abs_pct_error.describe())

print("MAPE:")
print(mape(arima_forecast.y, arima_forecast.yhat))
# MAPE:




########################################################################################
#      PROPHET (not expecting much here, y not seasonal in nature
########################################################################################

prophet = Prophet(seasonality_mode='multiplicative')

# simple prophet model
p = prophet.fit(train_df)
future = prophet.make_future_dataframe(periods=50, freq='M')

# make dates beginning of month
future['ds'] = future['ds'].values.astype('datetime64[M]')

forecast = p.predict(future)
fig = p.plot(forecast)
# a = add_changepoints_to_plot(fig.gca(), p, forecast)
fig.show()

fig2 = p.plot_components(forecast)
fig2.show()

pred_df = forecast.merge(df_model, how='left', on='ds')
pred_df.loc[:, 'error'] = pred_df.loc[:, 'y'] - pred_df.loc[:, 'yhat']

pred_df_test = pred_df[(~pred_df.y.isnull()) & (pred_df.ds >= test_min)]

mape_error = mape(pred_df_test.y, pred_df_test.yhat)
print(mape_error)
# baseline model, 42% error


