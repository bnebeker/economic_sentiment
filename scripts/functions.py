import pandas as pd
from pytrends.request import TrendReq
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from scripts.pytrends_functions import TrendReqDaily
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


def google_trends_data(kw_list=None, start_string='2004-01-01', end_string=None, resolution='REGION'):
    pytrend = TrendReq()
    date_list = pd.date_range(start_string, end_string, freq='1M') - pd.offsets.MonthBegin(1)

    data_list = []

    for index, dt in enumerate(date_list):
        next_dt = index + 1
        if next_dt < len(date_list):
            timeframe = dt.strftime('%Y-%m-%d') + ' ' + date_list[next_dt].strftime('%Y-%m-%d')
            print("DATE:", timeframe)

            pytrend.build_payload(kw_list, cat=0, timeframe=timeframe, geo='US', gprop='')

            # Interest by region
            # DMA ~ MAJOR CITY, REGION == STATE
            _df = pytrend.interest_by_region(
                resolution=resolution
            )  # metro level data ( you can change 'DMA' to 'CITY', 'REGION' )
            _df.reset_index(inplace=True)
            _df.loc[:, 'date'] = dt
            print(_df.head())
            print("\n\n")
            data_list.append(_df)

    df = pd.concat(data_list)
    print(df.shape)
    ## columns w/ spaces to _
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = map(str.lower, df.columns)

    return df


def google_trends_historical(kw_list=None, year_end='2020', month_end='05', geo="US"):
    pytrend = TrendReq()

    start_date = '2004-01-01'
    end_date = str(year_end) + '-' + str(month_end) + '-' + '01'
    timeframe = start_date + ' ' + end_date

    pytrend.build_payload(kw_list, cat=0, timeframe=timeframe, geo=geo, gprop='')

    df = pytrend.interest_over_time()
    df.reset_index(inplace=True)
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = map(str.lower, df.columns)

    return df


def google_trends_historical_daily(kw_list=None, year_end='2020', month_end='04', geo="US",
                                   year_start='2020', month_start='01', end_day='14'):
    pytrend = TrendReqDaily()

    start_date = str(year_start) + '-' + str(month_start) + '-' + '01'
    end_date = str(year_end) + '-' + str(month_end) + '-' + end_day
    timeframe = start_date + ' ' + end_date

    pytrend.build_payload(kw_list, cat=0, timeframe=timeframe, geo=geo, gprop='')

    df = pytrend.interest_over_time()
    df.reset_index(inplace=True)
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = map(str.lower, df.columns)

    return df


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def ts_evaluation(model=None, train_df=None, test_df=None, test_exog=None):
    forecast_df = test_df.copy()
    preds = model.forecast(test_df.shape[0], exog=test_exog)
    forecast_df.loc[:, 'yhat'] = preds
    forecast_df.loc[:, 'y_error'] = forecast_df.loc[:, 'y'] - forecast_df.loc[:, 'yhat']
    forecast_df.loc[:, 'y_pct_error'] = forecast_df.loc[:, 'y_error'] / forecast_df.loc[:, 'y']
    forecast_df.loc[:, 'y_abs_pct_error'] = abs(forecast_df.loc[:, 'y_pct_error'])
    # arima_forecast.loc[:, 'y_in_yhat_band'] = np.where((arima_forecast.y >= arima_forecast.yhat_lower) &
    # (arima_forecast.y <= arima_forecast.yhat_upper), 1, 0)

    print(forecast_df.y_abs_pct_error.describe())

    print("MAPE:")
    model_mape = mape(forecast_df.y, forecast_df.yhat)
    print(model_mape)

    train_preds = model.predict(full_results=True)

    out_train = pd.concat([train_df, train_preds], axis=1)
    out_train.columns = ['ds', 'y', 'yhat']

    # cut after a couple periods
    out_train = out_train[out_train.ds >= '2004-04-01']

    full_preds = out_train.append(forecast_df.loc[:, ['ds', 'y', 'yhat']])

    return full_preds, forecast_df, model_mape


def lag_features(df=None, col_list=None):
    for col in col_list:
        lag1 = col + '_lag1'
        lag2 = col + '_lag2'
        df.loc[:, lag1] = df.loc[:, col].shift(1)
        df.loc[:, lag2] = df.loc[:, col].shift(2)
        df.loc[:, col + '_trend_1'] = df.loc[:, col] - df.loc[:, lag1]
        df.loc[:, col + '_trend_2'] = df.loc[:, col] - df.loc[:, lag2]
        df.loc[:, col + '_pct_change_1'] = (df.loc[:, col] - df.loc[:, lag1]) / df.loc[:, lag1]
        df.loc[:, col + '_pct_change_2'] = (df.loc[:, col] - df.loc[:, lag2]) / df.loc[:, lag2]

    return df


def state_level_pred(state_df=None, model=None, features=None, target=None):
    state_preds = model.predict(state_df.loc[:, features])
    model_name = type(model).__name__
    pred_name = '{}_prediction_{}'.format(model_name, target)
    state_df.loc[:, pred_name] = state_preds
    # df_state_lim = state_df.loc[:, ['date', 'geo', pred_name]]

    return state_df, pred_name


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def model_eval(y_true=None, y_pred=None):
    model_r2 = r2_score(y_true, y_pred)
    model_rmse = mean_squared_error(y_true, y_pred)
    model_mape = mean_absolute_percentage_error(y_true, y_pred)

    print("MODEL R^2")
    print(model_r2)

    print("MODEL RMSE")
    print(model_rmse)

    return model_r2, model_rmse, model_mape
