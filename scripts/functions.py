import pandas as pd
from pytrends.request import TrendReq

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


def google_trends_data(kw_list=None, start_string='2004-01-01', end_string=None):
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
            _df = pytrend.interest_by_region(
                resolution='DMA'
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


def google_trends_historical(kw_list=None, year_end='2020', month_end='04', geo="US"):
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
