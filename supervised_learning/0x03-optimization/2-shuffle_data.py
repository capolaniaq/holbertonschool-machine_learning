#!/usr/bin/env python3
"""
Suffle twop matrices with same way
"""
import numpy as np


def shuffle_data(X, Y):
    """
    shuffles the data points in two matrices
    """
    X_shuffled = np.random.permutation(X)
    Y_shuffled = np.random.permutation(Y)
    return X_shuffled, Y_shuffled
