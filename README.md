# economic_sentiment

feature definitions
https://data.sca.isr.umich.edu/subset/subset.php

main targets: bus12 and umex which refer to business 
condition and the unemployment rate over the next year

google trends searches: economy, gross domestic product, employment growth, 
wages, wage growth, unemployment, unemployment insurance, uncertainty, 
stock market, instability

NOTE: google trends data is pulled one search term at a time (chunk size = 1 in data script)
this is because the trends data is normalized 0-100 against all search terms in the query
so any data that is at its all time peak should be 100, but if it's lower than other search terms
in the batch, it will be less
