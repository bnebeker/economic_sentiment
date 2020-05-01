from scripts.functions import google_trends_data, google_trends_historical
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

# BY REGION
chunk_size = 1
for i in range(0, len(kw_list), chunk_size):
    list_subset = kw_list[i:i + chunk_size]
    print(list_subset)
    _df = google_trends_data(kw_list=list_subset, end_string='2020-05-01')

    if i == 0:
        output_df = _df.copy()
    else:
        output_df = output_df.merge(
            _df,
            how='left',
            on=['geoname', 'date']
        )

# move date to the front
output_df = output_df[['date'] + [col for col in output_df.columns if col != 'date']]

output_df.to_csv(
    './data/google_trends.csv.tar.bz2',
    compression='bz2',
    index=False
)
