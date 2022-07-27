#!/usr/bin/env python3

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd

from_file = __import__('2-from_file').from_file

df = from_file('/content/gdrive/My Drive/Colab Notebooks/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df.drop('Weighted_Price', inplace=True, axis=1)

df.rename(columns={'Timestamp': 'Date'}, inplace=True)

df['Date'] = df['Date'].apply(lambda x: datetime.fromtimestamp(x))

df = df[df['Date'] >= '2017-01-01']

df.set_index('Date', inplace=True)

df['Close'].ffill(inplace=True)

df['High'].fillna(value=df['Close'], inplace=True)
df['Low'].fillna(value=df['Close'], inplace=True)
df['Open'].fillna(value=df['Close'], inplace=True)

df['Volume_(BTC)'].fillna(value=0, inplace=True)
df['Volume_(Currency)'].fillna(value=0, inplace=True)

new_df = pd.DataFrame()

new_df['High'] = df['High'].resample('D').max()
new_df['Low'] = df['Low'].resample('D').min()
new_df['Open'] = df['Open'].resample('D').mean()
new_df['Close'] = df['Close'].resample('D').mean()
new_df['Volume_(BTC)'] = df['Volume_(BTC)'].resample('D').sum()
new_df['Volume_(Currency)'] = df['Volume_(Currency)'].resample('D').sum()

new_df.plot()
plt.show()