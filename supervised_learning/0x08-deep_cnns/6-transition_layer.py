#!/usr/bin/env python3
"""
compression as used in DenseNet-C
"""

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    X is the output from the previous layer
    nb_filters is an integer representing the number of filters in X
    compression is the compression factor for the transition layer
    """
    init = K.initializers.he_normal()
    batch_norm = K.layers.BatchNormalization()(X)
    activation = K.layers.Activation('relu')(batch_norm)
    convol = K.layers.Conv2D(filters=(nb_filters * compression), kernel_size=1,
                             padding='same',
                             kernel_initializer=init)(activation)
    avg = K.layers.AveragePooling2D(pool_size=(2, 2), strides=2)(convol)
    return avg, nb_filters * compression
