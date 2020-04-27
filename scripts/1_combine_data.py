import pandas as pd

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

df_s = pd.read_csv(
    './data/umichigan_allsentiment.csv.tar.bz2',
    compression='bz2'
)

df_s.loc[:, 'day'] = 1
df_s.loc[:, 'date'] = pd.to_datetime(df_s[['year', 'month', 'day']])

df_t = pd.read_csv(
    './/data/google_trends.csv.tar.bz2',
    compression='bz2'
)

