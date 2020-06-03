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


def create_dataset(data=None, features=None):
    df = lag_features(df=data, col_list=features)

    # fill nulls with zeros
    df.fillna(0, inplace=True)

    # replace inf with 1
    df.replace([np.inf], 1, inplace=True)

    # drop first two rows, no lagged features
    df.drop(df.index[0:2], inplace=True)

    return df


df = create_dataset(data=df, features=col_list)

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
    './data/prepared/prepared_data_full_us.csv.tar.bz2',
    compression='bz2',
    index=False
)

###########################################################################################################
#      EARNINGS DATA
###########################################################################################################
df_earn_n = pd.read_csv(
    './data/earnings_national.csv.tar.bz2',
    compression='bz2'
)

df_earn_n.loc[:, 'day'] = 1
df_earn_n.loc[:, 'date'] = pd.to_datetime(df_earn_n[['year', 'month', 'day']])

df_earn_n = df_t.merge(
    df_earn_n,
    how='left',
    on='date'
)

df_earn_n = create_dataset(data=df_earn_n, features=col_list)

# bring target values to the front
cols = df_earn_n.columns.tolist()
cols.insert(0, cols.pop(cols.index('date')))
cols.insert(1, cols.pop(cols.index('privatenatlemp')))
cols.insert(2, cols.pop(cols.index('weeklynatlearn')))
df_earn_n = df_earn_n.reindex(columns=cols)

df_earn_n.to_csv(
    './data/prepared/earnings_national.csv.tar.bz2',
    index=False,
    compression='bz2'
)

###########################################################################################################
#      BY STATE
###########################################################################################################
df_t = pd.read_csv(
    './data/google_trends.csv.tar.bz2',
    compression='bz2'
)

cols = df_t.columns.tolist()
cols.insert(0, cols.pop(cols.index('date')))
cols.insert(1, cols.pop(cols.index('geo')))
df_t = df_t.reindex(columns=cols)

col_list = df_t.columns[2:]

df_t = create_dataset(data=df_t, features=col_list)

print(df_t.shape)

df_t.to_csv(
    './data/prepared/prepared_data_by_state.csv.tar.bz2',
    compression='bz2',
    index=False
)


###########################################################################################################
#      EARNINGS DATA BY STATE
###########################################################################################################
df_earn_st = pd.read_csv(
    './data/earnings_state.csv.tar.bz2',
    compression='bz2'
)

df_earn_st.loc[:, 'day'] = 1
df_earn_st.loc[:, 'date'] = pd.to_datetime(df_earn_st[['year', 'month', 'day']])

## map st to geo
## join with google trends data

df_earn_st = create_dataset(data=df_earn_st, features=col_list)
