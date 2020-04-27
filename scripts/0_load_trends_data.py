from scripts.functions import google_trends_data
import pandas as pd

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

kw_list = [
    "unemployment",
    "jobs",
    "economy",
    "gross domestic product",
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
    "instability"
]

df = google_trends_data(kw_list=kw_list, end_string='2020-05-01')

df.to_csv(
    './data/google_trends.csv.tar.bz2',
    compression='bz2',
    index=False
)
