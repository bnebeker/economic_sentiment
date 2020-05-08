from scripts.functions import google_trends_historical
import pandas as pd

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

kw_list = [
    "unemployment",
    "jobs",
    "economy",
    "gross domestic product",
    "instability",
    "GDP",
    "wages",
    "wage growth",
    "employment growth",
    "economic growth",
    "unemployment insurance",
    "uncertainty",
    "stock market",
    # "pandemic",
    # "coronavirus",
    # "covid",
    "stocks"
]

# reset chunk size to 1
# values are normalized against the rest of the search items
# makes more sense to have them only normalized against themselves

# ENTIRE US, INSTEAD OF BY REGION
chunk_size = 1
for i in range(0, len(kw_list), chunk_size):
    list_subset = kw_list[i:i + chunk_size]
    print(list_subset)
    _df = google_trends_historical(kw_list=list_subset)
    _df.drop('ispartial', inplace=True, axis=1)

    if i == 0:
        output_df_all_us = _df.copy()
    else:
        output_df_all_us = output_df_all_us.merge(
            _df,
            how='left',
            on=['date']
        )

# move date to the front
output_df_all_us = output_df_all_us[['date'] + [col for col in output_df_all_us.columns if col != 'date']]

output_df_all_us.to_csv(
    './data/google_trends_full_us.csv.tar.bz2',
    compression='bz2',
    index=False
)


state_list = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
]

# BY REGION - UPDATED

output_df = pd.DataFrame()

start_date = '2004-01-01'
year_end = 2020
month_end = '04'
end_date = str(year_end) + '-' + str(month_end) + '-' + '01'

date_list = pd.date_range(start_date, end_date, freq='1M') - pd.offsets.MonthBegin(1)

for geo in state_list:
    print(geo)
    for i in range(0, len(kw_list)):
        list_subset = kw_list[i:i + 1]

        print(list_subset)
        _df = google_trends_historical(
            kw_list=list_subset,
            geo='US-{}'.format(geo)
        )

        if _df.empty:
            col = list_subset[0].replace(' ', '_')

            _df.loc[:, 'date'] = date_list
            _df.loc[:, col] = 0
            _df.loc[:, 'ispartial'] = False
            _df.drop('index', axis=1, inplace=True)

        _df.drop('ispartial', inplace=True, axis=1, errors='ignore')
        _df.loc[:, 'geo'] = geo

        if i == 0:
            state_df = _df.copy()
        else:
            state_df = state_df.merge(
                _df,
                how='left',
                on=['date', 'geo']
            )

    output_df = output_df.append(state_df, ignore_index=True)

# move date to the front
output_df = output_df[['date'] + [col for col in output_df.columns if col != 'date']]

output_df.to_csv(
    './data/google_trends.csv.tar.bz2',
    compression='bz2',
    index=False
)
