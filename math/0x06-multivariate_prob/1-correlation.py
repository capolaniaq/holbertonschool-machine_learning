#!/usr/bin/env python3
"""
Calculate the correlation matrix of a set of variables
"""

import numpy as np


def correlation(C):
    """
    C is a numpy.ndarray of shape (d, d) containing a covariance matrix
    d is the number of dimensions
    Return correlation matrix
    """
    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2 or (C.shape[0] != C.shape[1]):
        raise ValueError("C must be a 2D square matrix")
    d, _ = C.shape
    cor = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            cor[i][j] = C[i][j] / (np.sqrt(C[i][i] * C[j][j]))
    return cor
