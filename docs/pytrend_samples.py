import pandas as pd
from pytrends.request import TrendReq
pytrend = TrendReq()

pytrend.build_payload(kw_list=['economy', 'GDP'])
# Interest by Region
df = pytrend.interest_by_region()
df.head(10)

keywords = pytrend.suggestions(keyword='economy')
df2 = pd.DataFrame(keywords)
df2.drop(columns='mid')


pytrend.build_payload(kw_list=['economy'])
# Related Queries, returns a dictionary of dataframes
related_queries = pytrend.related_queries()
related_queries.values()

related_topic = pytrend.related_topics()
related_topic.values()
