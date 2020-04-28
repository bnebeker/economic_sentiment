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

# get lag and trend
# not good process but small df, fine for now
col_list = df.columns[1:-2]

for col in col_list:
    new_col = col + '_lag'
    df.loc[:, new_col] = df.loc[:, col].shift(1)
    df.loc[:, col + '_trend'] = df.loc[:, col] - df.loc[:, new_col]
    df.loc[:, col + '_pct_change'] = (df.loc[:, col] - df.loc[:, new_col]) / df.loc[:, new_col]

# fill nulls with zeros
df.fillna(0, inplace=True)

# drop first row, no lagged features
df.drop(df.index[0], inplace=True)

df.to_csv(
    './data/prepared_data_full_us.csv.tar.bz2',
    compression='bz2',
    index=False
)
