#!/usr/bin/env python3
"""
Lenet-5 architecture using tensorflow and keras
"""

import tensorflow.keras as K


def lenet5(X):
    """
    X is a K.Input of shape (m, 28, 28, 1) containing the input
    images for the network
        m is the number of images
    The model should consist of the following layers in order:
        Convolutional layer with 6 kernels of shape 5x5 with same padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Convolutional layer with 16 kernels of shape 5x5 with valid padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Fully connected layer with 120 nodes
        Fully connected layer with 84 nodes
        Fully connected softmax output layer with 10 nodes
    """
    initializer = K.initializers.VarianceScaling(scale=2.0)

    input = K.Input(shape=(28, 28, 1))

    convol_1 = K.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same',
                               activation='relu',
                               kernel_initializer=initializer)(input)

    sub_1 = K.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(convol_1)

    convol_2 = K.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid',
                               activation='relu',
                               kernel_initializer=initializer)(sub_1)

    sub_2 = K.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(convol_2)

    flatten = K.layers.Flatten()(sub_2)

    dense_1 = K.layers.Dense(units=120, activation='relu',
                             kernel_initializer=initializer)(flatten)
    dense_2 = K.layers.Dense(units=84, activation='relu',
                             kernel_initializer=initializer)(dense_1)
    output = K.layers.Dense(units=10, activation='softmax',
                            kernel_initializer=initializer)(dense_2)

    model = K.Model(inputs=input, outputs=output)

    adam = K.optimizers.Adam()
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model
