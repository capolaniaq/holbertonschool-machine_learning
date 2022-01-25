#!/usr/bin/env python3
"""
build a neural network with the Keras library
"""

import tensorflow.keras as K


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
    input = K.Input(shape=(nx, ))
    L2 = K.regularizers.L2(lambtha)
    for i in range(len(layers)):
        if i == 0:
            x = K.layers.Dense(layers[i], activation=activations[i],
                               kernel_regularizer=L2)(input)
        else:
            d = K.layers.Dropout(1 - keep_prob)(x)
            x = K.layers.Dense(layers[i], activation=activations[i],
                               kernel_regularizer=L2)(d)
    model = K.Model(inputs=input, outputs=x)
    return model
