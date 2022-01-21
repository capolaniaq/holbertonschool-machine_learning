#!/usr/bin/env python3
"""
Gradient descent with L2 regularization.
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Gradient Descent with L2 regularization
    Y correct output
    weights = is a dictionary with Ws and bs values for each layers
    cache = is a dictionary with each As values for layer
    alpha = is a learning rate
    Lambtha = is a L2 parameter
    L = Number of layers
    """
    m = Y.shape[1]
    for i in range(L, 0, -1):
        if i == L:
            dZ = cache['A' + str(i)] - Y
            dW = 1 / m * np.matmul(cache['A' + str(i - 1)], dZ.T) + lambtha / m * weights['W' + str(i)]
            db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        else:
            dZ = np.matmul(weights['W' + str(i + 1)].T, dZ) * (1 - np.power(cache['A' + str(i)], 2))
            dW = 1 / m * np.matmul(cache['A' + str(i - 1)], dZ.T) + lambtha / m * weights['W' + str(i)]
            db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        weights['W' + str(i)] = weights['W' + str(i)] - alpha * dW
        weights['b' + str(i)] = weights['b' + str(i)] - alpha * db
    return weights, cache
