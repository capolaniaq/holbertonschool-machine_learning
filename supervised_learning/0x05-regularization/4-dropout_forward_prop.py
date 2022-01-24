#!/usr/bin/env python3
"""
Forward propagation with dropout
"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    forward prop with dropout
    X is a numpy.ndarray of shape (nx, m) containing the input
    data for the network
        nx is the number of input features
        m is the number of data points
    weights is a dictionary of the weights and biases of the
    neural network
    L the number of layers in the network
    keep_prob is the probability that a node will be kept
    All layers except the last should use the tanh activation function
    The last layer should use the softmax activation function
    """
    cache = {}
    cache['A0'] = X
    for i in range(L):
        b = weights['b' + str(i + 1)]
        W = weights['W' + str(i + 1)]
        x = cache['A' + str(i)]
        key = 'A{}'.format(i + 1)
        z = np.matmul(W, x) + b
        if i == L - 1:
            cache[key] = np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)
        else:
            cache[key] = np.tanh(z)
            D = 'D' + str(i + 1)
            cache[D] = np.random.rand(cache[key].shape[0], cache[key].shape[1])
            cache[D] = np.where(cache[D] < keep_prob, 1, 0)
            cache[key] = cache[key] * cache['D' + str(i + 1)]
            cache[key] = cache[key] / keep_prob
    return cache
