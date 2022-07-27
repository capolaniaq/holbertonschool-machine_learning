#!/usr/bin/env python3

import pandas as pd

from_file = __import__('2-from_file').from_file

df1 = from_file('/content/gdrive/My Drive/Colab Notebooks/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df2 = from_file('/content/gdrive/My Drive/Colab Notebooks/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', ',')

df1.set_index('Timestamp', drop=True, inplace=True)
df2.set_index('Timestamp', drop=True, inplace=True)

df = pd.concat([df1.loc['1417411980':'1417417980'], df2.loc['1417411980':'1417417980']], keys=['bitstamp', 'coinbase'])

df = df.swaplevel(0, 1)

df = df.sort_index()

print(df)