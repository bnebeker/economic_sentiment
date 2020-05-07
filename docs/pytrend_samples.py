import pandas as pd
from pytrends.request import TrendReq
pytrend = TrendReq()

pytrend.build_payload(kw_list=['economy', 'GDP'])
# Interest by Region
df = pytrend.interest_by_region()
df.head(10)

df = pytrend.interest_over_time()

keywords = pytrend.suggestions(keyword='economy')
df2 = pd.DataFrame(keywords)
df2.drop(columns='mid')


pytrend.build_payload(kw_list=['economy'])
# Related Queries, returns a dictionary of dataframes
related_queries = pytrend.related_queries()
related_queries.values()

related_topic = pytrend.related_topics()
related_topic.values()



start_date = '2004-01-01'
end_date = '2010-01-01'
timeframe = start_date + ' ' + end_date

kw_list = ['economy', 'gdp']

pytrend.build_payload(kw_list, cat=0, timeframe=timeframe, geo='US-AZ', gprop='')

df = pytrend.interest_over_time()
df.reset_index(inplace=True)
df.columns = df.columns.str.replace(' ', '_')
df.columns = map(str.lower, df.columns)
