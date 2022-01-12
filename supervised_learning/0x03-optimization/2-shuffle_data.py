#!/usr/bin/env python3
"""
Suffle two matrices with same way
"""
import numpy as np


def shuffle_data(X, Y):
    """
    shuffles the data points in two matrices
    """
    m = X.shape[0]
    permutation = np.random.permutation(m)
    print(permutation)
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :]
    return shuffled_X, shuffled_Y
