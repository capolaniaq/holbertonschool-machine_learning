#!/usr/bin/env python3
"""
build a neural network with the Keras library
"""

import tensorflow.keras as Keras


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    nx is the number of input features to the network
    layers is a list containing the number of nodes in each
    layer of the network
    activations is a list containing the activation functions
    used for each layer of the network
    lambtha is the L2 regularization parameter
    keep_prob is the probability that a node will be kept for dropout
    You are not allowed to use the Sequential class
    """
    input = Keras.Input(shape=(nx, ), name='input_1')
    L2 = Keras.regularizers.L2(lambtha)
    for i in range(len(layers)):
        if i == 0:
            x = Keras.layers.Dense(layers[i], activation=activations[i],
                                   kernel_regularizer=L2, name='dense')(input)
        elif i == len(layers) - 1:
            x = Keras.layers.Dropout(1 - keep_prob)(x)
            output = Keras.layers.Dense(layers[i], activation=activations[i],
                                        kernel_regularizer=L2, name='dense_' + str(i))(x)
        else:
            x = Keras.layers.Dropout(1 - keep_prob)(x)
            x = Keras.layers.Dense(layers[i], activation=activations[i],
                                   kernel_regularizer=L2, name='dense_' + str(i))(x)
    model = Keras.Model(inputs=input, outputs=output)
    return model
