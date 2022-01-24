#!/usr/bin/env python3
"""
Create layer with dropout
"""

import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    prev is a tensor containing the output of the previous layer
    n is the number of nodes the new layer should contain
    activation is the activation function that should be used on
    the layer
    keep_prob is the probability that a node will be kept
    Returns: the output of the new layer
    """
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                        mode=("fan_avg"))
    dropout = tf.layers.Dropout(rate=(1 - keep_prob))
    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=initializer,
                            kernel_regularizer=dropout,
                            name='layer')
    return layer(prev)
