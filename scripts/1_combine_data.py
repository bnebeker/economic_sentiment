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
    './data/google_trends_full_us.csv.tar.bz2',
    compression='bz2'
)

df_t.loc[:, 'date'] = pd.to_datetime(df_t.loc[:, 'date'])

print("SENTIMENT DATA...")
print(df_s.shape)
print(df_s.head())

print("")
print("TRENDS DATA...")
print(df_t.shape)
print(df_t.head())

df_s = df_s.loc[:, ['date', 'bus12_r_all', 'umex_r_all']]
df_s.columns = ['date', 'target_bus12', 'target_umex']

df = df_t.merge(
    df_s,
    how='left',
    on='date'
)

df.to_csv(
    './data/combined_data_full_us.csv.tar.bz2',
    compression='bz2',
    index=False
)

