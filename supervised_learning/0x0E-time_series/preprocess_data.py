#!/usr/bin/env python3
"""
Module for preprocessing data.
"""

import numpy as np
import matplotlib.pyplot as plt

import os
from datetime import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

# Inport data
df = pd.read_csv("coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv")

# Convert timestamp to datetime
df['Timestamp'] = df['Timestamp'].apply(lambda x: datetime.fromtimestamp(x))

# full Nan values
df = df.fillna(method='ffill', axis=0)

df.isnull().sum()

# Join the rows per hour
df = df[7::60]

# Make only one column and Timestamp for index
df = df[['Timestamp', 'Close']]
df = df.set_index('Timestamp')


def split_data(data, split_size=0.8):
    """
    Split data into train and test data
    """
    n = len(data)
    train_df = data[:int(n * split_size)]
    test_df = data[int(n * split_size):]
    return train_df, test_df

train_df, test_df = split_data(df)

def normalize_data(train_data, test_data):
    """
    Normalize data between 0 and 1
    """
    train_mean = train_data.mean()
    train_std = train_data.std()

    train_data = (train_data - train_mean) / train_std
    test_data = (test_data - train_mean) / train_std
    return train_data, test_data

train_df, test_df = normalize_data(train_df, test_df)
