from scripts.functions import google_trends_data
import pandas as pd

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

# larger lists returning 400 error, breaking kw list into smaller pieces
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
    "pandemic",
    "coronavirus",
    "covid",
    "stocks"
]

chunk_size = 5
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

output_df.to_csv(
    './data/google_trends.csv.tar.bz2',
    compression='bz2',
    index=False
)
