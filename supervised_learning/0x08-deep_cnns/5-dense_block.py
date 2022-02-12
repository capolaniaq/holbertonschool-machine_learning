#!/usr/bin/env python3
"""
DenseNet-Dense Block
"""

import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    X is the output from the previous layer
    nb_filters is an integer representing the number of filters in X
    growth_rate is the growth rate for the dense block
    layers is the number of layers in the dense block
    You should use the bottleneck layers used for DenseNet-B
    All weights should use he normal initialization
    All convolutions should be preceded by Batch Normalization
    and a rectified linear activation (ReLU), respectively
    Returns: The concatenated output of each layer within the
    Dense Block and the number of filters within the concatenated
    outputs, respectively
    """
    init = K.initializers.HeNormal()

    for i in range(layers):
        batch_norm = K.layers.BatchNormalization()(X)
        activation = K.layers.Activation('relu')(batch_norm)
        convol = K.layers.Conv2D(filters=4*growth_rate, kernel_size=(1, 1),
                                 padding='same',
                                 kernel_initializer=init)(activation)
        batch_norm_1 = K.layers.BatchNormalization()(convol)
        activation_1 = K.layers.Activation('relu')(batch_norm_1)
        convol_1 = K.layers.Conv2D(filters=growth_rate, kernel_size=(3, 3),
                                   padding='same',
                                   kernel_initializer=init)(activation_1)
        X = K.layers.concatenate([X, convol_1])
        nb_filters += growth_rate
    return X, nb_filters
