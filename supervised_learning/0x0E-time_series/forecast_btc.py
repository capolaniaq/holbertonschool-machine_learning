#!/usr/bin/env python3
"""
Make a time series forecasting model using LSTM.
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


INPUT_WIDTH = 24
LABEL_WIDTH = 1



def split_window(data):
    """
    Function that make a window of data
    """
    input_slice = slice(0, INPUT_WIDTH)
    output_slice = slice(INPUT_WIDTH, None)

    inputs = data[:, input_slice, :]
    outputs = data[:, output_slice, :]

    return inputs, outputs

def make_dataset(data):
    """
    Make dataset for training and testing
    and use for tensorflow
    """
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data= data,
        targets = None,
        sequence_length = INPUT_WIDTH + LABEL_WIDTH,
        sequence_stride = 1,
        shuffle = True,
        batch_size = 32
    )

    ds = ds.map(split_window)
    return ds

train_df = make_dataset(train_df)
test_df = make_dataset(test_df)

def compile_and_fit(model, train_ds, test_ds, patience=1):
    """
    Function that compile and fit the model
    """
    model.compile(
        loss= tf.losses.MeanAbsoluteError(),
        optimizer= tf.optimizers.Adam(),
        metrics=[tf.metrics.MeanAbsoluteError()]
    )


    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience= patience,
        mode = 'min'
    )

    history = model.fit(
        train_ds,
        epochs=5,
        validation_data= test_ds,
        callbacks= [early_stopping]
    )

    return history

simple_model = tf.keras.models.Sequential([
                                    tf.keras.layers.LSTM(32, return_sequences=False),
                                    tf.keras.layers.Dense(1)
                                    ])

history = compile_and_fit(simple_model, train_df, test_df)
