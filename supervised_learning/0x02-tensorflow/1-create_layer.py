#!/usr/bin/env python
"""
Create a first layer of a new project.
"""
from tensorflow.keras import initializers
import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Create a layer function.
    """
    inicializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(n, activation=activation, kernel_initializer=inicializer, name='layer')
    return layer(prev)
