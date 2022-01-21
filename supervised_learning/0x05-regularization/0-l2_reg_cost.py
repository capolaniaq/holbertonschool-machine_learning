#!/usr/bin/env python3
"""
Calculates the cost of a neural network with L2 regularization
"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculated the cost with L2 regularization
    """
    weight = 0
    for i in range(L):
        weight += np.sum(np.square(weights['W' + str(i + 1)]))
    cost_l2 = cost + (lambtha / (2 * m)) * weight
    return cost_l2
