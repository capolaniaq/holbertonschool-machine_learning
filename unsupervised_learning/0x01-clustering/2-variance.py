#!/usr/bin/env python3
"""
variance for K clusters
"""

import numpy as np


def variance(X, C):
    """
    X is a numpy.ndarray of shape (n, d) containing the data set
    C is a numpy.ndarray of shape (k, d) containing the centroid
    means for each cluster
    You are not allowed to use any loops
    Returns: var, or None on failure
    var is the total variance
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(C) is not np.ndarray or len(C.shape) != 2:
        return None
    n, d = X.shape
    k, d = C.shape
    C_extend = C.reshape(k, 1, d)
    dist = np.sqrt(np.sum(np.square(C_extend - X), axis=2))
    min_dist = np.amin(dist, axis=0)
    variance = np.sum(np.square(min_dist))
    return variance
