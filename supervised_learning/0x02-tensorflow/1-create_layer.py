#!/usr/bin/env python3
"""
Create a first layer of a new project.
"""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
    Create a layer function.
    """
    inicializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=inicializer, name='layer')
    return layer(prev)
