#!/usr/bin/env python3
"""
Build an identity block
"""

import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    A_prev is the output from the previous layer
    filters is a tuple or list containing F11, F3, F12, respectively:
        F11 is the number of filters in the first 1x1 convolution
        F3 is the number of filters in the 3x3 convolution
        F12 is the number of filters in the second 1x1 convolution
    """
    F11, F3, F12 = filters
    initializer = K.initializers.HeNormal()
    convol = K.layers.Conv2D(filters=F11, kernel_size=(1, 1), padding='same',
                             activation='relu',
                             kernel_initializer=initializer)(A_prev)
    batch_norm = K.layers.BatchNormalization()(convol)
    activation = K.layers.Activation('relu')(batch_norm)
    convol_1 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), padding='same',
                               activation='relu',
                               kernel_initializer=initializer)(activation)
    batch_norm_1 = K.layers.BatchNormalization()(convol_1)
    activation_1 = K.layers.Activation('relu')(batch_norm_1)
    convol_2 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), padding='same',
                               activation='relu',
                               kernel_initializer=initializer)(activation_1)
    batch_norm_2 = K.layers.BatchNormalization()(convol_2)
    add = K.layers.Add()([batch_norm_2, A_prev])
    activation_2 = K.layers.Activation('relu')(add)

    return activation_2
