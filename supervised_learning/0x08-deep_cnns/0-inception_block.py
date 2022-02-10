#!/usr/bin/env python3
"""
Inception Block
"""

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    A_prev is the output from the previous layer
    filters is a tuple or list containing F1, F3R, F3,F5R, F5, FPP,
    respectively:
    F1 is the number of filters in the 1x1 convolution
    F3R is the number of filters in the 1x1 convolution
    before the 3x3 convolution
    F3 is the number of filters in the 3x3 convolution
    F5R is the number of filters in the 1x1 convolution
    before the 5x5 convolution
    F5 is the number of filters in the 5x5 convolution
    FPP is the number of filters in the 1x1 convolution
    after the max pooling
    (Note : The output shape after the max pooling layer is
    outputshape = math.floor((inputshape - 1) / strides) + 1)
    """
    F1, F3R, F3,F5R, F5, FPP = filters
    conv1 = K.layers.Conv2D(F1, kernel_size=(1, 1), padding='same',
                            activation='relu')(A_prev)

    conv2 = K.layers.Conv2D(F3R, kernel_size=(1, 1), padding='same',
                            activation='relu')(A_prev)

    conv3 = K.layers.Conv2D(F3, kernel_size=(3, 3), padding='same',
                            activation='relu')(conv2)

    conv4 = K.layers.Conv2D(F5R, kernel_size=(1, 1), padding='same',
                            activation='relu')(A_prev)

    conv5 = K.layers.Conv2D(F5, kernel_size=(5, 5), padding='same',
                            activation='relu')(conv4)

    maxpool = K.layers.MaxPooling2D((3, 3), strides=(1, 1),
                                    padding='same')(A_prev)

    maxpool_conv = K.layers.Conv2D(FPP, kernel_size=(1, 1), padding='same',
                                   activation='relu')(maxpool)

    concatenation = K.layers.concatenate([conv1, conv3, conv5, maxpool_conv],
                                         axis=3)

    return concatenation
