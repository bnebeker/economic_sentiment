from scripts.functions import google_trends_daily
import pandas as pd

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

kw_list = [
    "child care",
    "rain",
    "clouds",
    "unemployment benefits",
    "disney plus"
]

chunk_size = 1
for i in range(1, len(kw_list), chunk_size):
    list_subset = kw_list[i:i + chunk_size]
    print(list_subset)
    _df = google_trends_daily(kw_list=list_subset, end_string='2020-05-01')

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
    './child_care/google_trends_childcare.csv.tar.bz2',
    compression='bz2',
    index=False
)

# output_df.to_csv(
#     './child_care/google_trends_childcare.csv',
#     index=False
# )
