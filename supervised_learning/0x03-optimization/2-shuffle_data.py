#!/usr/bin/env python3
"""
Suffle two matrices with same way
"""
import numpy as np


def shuffle_data(X, Y):
    """
    shuffles the data points in two matrices
    """
    if (X.shape[0] != Y.shape[0]):
        return None
    p = np.random.permutation(X.shape[0])
    return X[p], Y[p]
