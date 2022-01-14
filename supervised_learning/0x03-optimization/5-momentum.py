#!/usr/bin/env python3
"""
updates the variables of a neural network using gradient descent with
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    update the variables of a neural network using gradient descent with
    the momentum optimization algorithm
    :param alpha: learning rate
    :param beta1: momentum weight
    :param var: numpy.ndarray
    :param grad: numpy.ndarray
    :param v: numpy.ndarray
    :return: the updated variable and the velocity
    """
    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v
    return var, v
