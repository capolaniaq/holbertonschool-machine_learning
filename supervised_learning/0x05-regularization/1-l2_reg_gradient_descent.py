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
    The neural network uses tanh activations on each layer except the last, which uses a softmax activation
    The weights and biases of the network should be updated in place
    """
    m = Y.shape[1]
    weights_cop = weights.copy()
    for i in range(L, 0, -1):
        A = cache['A' + str(i)]
        if i == L:
            dz = A - Y
        else:
            dz = np.matmul(weights_cop['W' + str(i + 1)].T, dz) * (1 - ( A * A))
        l2 = (lambtha * weights_cop['W' + str(i)]) / m
        dw = 1 / m * np.matmul(dz, cache['A' + str(i - 1)].T) + l2
        db = 1 / m * np.sum(dz, axis=1, keepdims=True)
        weights["W" + str(i)] = weights_cop['W' + str(i)] - alpha * dw
        weights["b" + str(i)] = weights_cop['b' + str(i)] - alpha * db
