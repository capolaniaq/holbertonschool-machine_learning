#!/usr/bin/env python3
"""
Adam Optimization
"""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    alpha is the learning rate
    beta1 is the weight used for the first moment
    beta2 is the weight used for the second moment
    epsilon is a small number to avoid division by zero
    var is a numpy.ndarray containing the variable to be updated
    grad is a numpy.ndarray containing the gradient of var
    v is the previous first moment of var
    s is the previous second moment of var
    t is the time step used for bias correction
    Returns: the updated variable, the new first moment,
    and the new second moment, respectively
    """
    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * grad**2
    v_correction = v / (1 - beta1 ** t)
    s_correction = s / (1 - beta2 ** t)
    var = var - alpha * v_correction / (np.sqrt(s_correction) + epsilon)
    return var, v, s
