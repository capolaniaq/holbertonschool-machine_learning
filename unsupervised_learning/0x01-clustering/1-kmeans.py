#!/usr/bin/env python3
"""
K-Means Clustering
"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    X is a numpy.ndarray of shape (n, d) containing the dataset
    n is the number of data points
    d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters
    iterations is a positive integer containing the maximum number
    of iterations that should be performed
    Returns: C, clss, or None, None on failure
    C is a numpy.ndarray of shape (k, d) containing the centroid
    means for each cluster
    clss is a numpy.ndarray of shape (n,) containing the index of
    the cluster in C that each data point belongs to
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(k) is not int or k <= 0 or k > X.shape[0]:
        return None, None
    if type(iterations) is not int and iterations <= 0:
        return None, None
    n, d = X.shape
    low = np.amin(X, axis=0)
    high = np.amax(X, axis=0)
    C = np.random.uniform(low=low, high=high, size=(k, d))
    for i in range(iterations):
        C_old = C.copy()
        cls = np.zeros(n)
        c_extend = C[:, np.newaxis]
        dist = np.sqrt(np.sum((X - c_extend) ** 2, axis=2))
        cls = np.argmin(dist, axis=0)
        for j in range(k):
            if X[cls == j].size == 0:
                C[j] = np.random.uniform(low=low, high=high, size=(1, d))
            else:
                C[j] = np.mean(X[cls == j], axis=0)
        c_extend = C[:, np.newaxis]
        dist = np.sqrt(np.sum((X - c_extend) ** 2, axis=2))
        cls = np.argmin(dist, axis=0)
        if np.array_equal(C, C_old):
            break
    return C, cls
