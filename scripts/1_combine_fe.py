import pandas as pd
import numpy as np
from scripts.functions import lag_features
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

df = lag_features(df=df, col_list=col_list)

# fill nulls with zeros
df.fillna(0, inplace=True)

# replace inf with 1
df.replace([np.inf], 1, inplace=True)

# drop first two rows, no lagged features
df.drop(df.index[0:2], inplace=True)

# bring target values to the front
cols = df.columns.tolist()
cols.insert(0, cols.pop(cols.index('date')))
cols.insert(1, cols.pop(cols.index('target_bus12')))
cols.insert(2, cols.pop(cols.index('target_umex')))
df = df.reindex(columns=cols)

print(df.shape)
output = df[df.target_bus12 != 0]
print(output.shape)
output.to_csv(
    './data/prepared_data_full_us.csv.tar.bz2',
    compression='bz2',
    index=False
)

## BY STATE
df_t = pd.read_csv(
    './data/google_trends.csv.tar.bz2',
    compression='bz2'
)

cols = df_t.columns.tolist()
cols.insert(0, cols.pop(cols.index('date')))
cols.insert(1, cols.pop(cols.index('geo')))
df_t = df_t.reindex(columns=cols)

col_list = df_t.columns[2:]

df_t = lag_features(df=df_t, col_list=col_list)

# fill nulls with zeros
df_t.fillna(0, inplace=True)

# replace inf with 1
df_t.replace([np.inf], 1, inplace=True)

# drop first two rows, no lagged features
df_t.drop(df_t.index[0:2], inplace=True)

print(df_t.shape)

df_t.to_csv(
    './data/prepared_data_by_state.csv.tar.bz2',
    compression='bz2',
    index=False
)
