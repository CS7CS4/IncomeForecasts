import pandas as pd

data = pd.read_csv('data_drop_c.csv', encoding = 'utf-8')


def _(val):
    if val == 'San Jose':
        val = 0
        print(val)
    elif val == 'New York':
        val = 1
    elif val == 'San Francisco':
        val = 2
    elif val == 'California':
        val = 3
    return val

data[u'location_filter'] = data[u'location_filter'].astype(str)
data[u'location_filter'] = data[u'location_filter'].apply(lambda x:_(x))

# data.loc[(data['location_filter'] == "San Jose"), "location_filter"] = 0


data.to_csv('data_drop_c.csv', index = None, encoding = 'utf-8')