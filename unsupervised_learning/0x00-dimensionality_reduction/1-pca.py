#!/usr/bin/env python3
"""
PCA on a dataset
"""
import numpy as np


def pca(X, ndim):
    """
    X is a numpy.ndarray of shape (n, d) where:
    n is the number of data points
    d is the number of dimensions in each point
    ndim is the new dimensionality of the transformed X
    Returns: T, a numpy.ndarray of shape (n, ndim)
    containing the transformed version of X
    """
    U, S, V = np.linalg.svd(X)
    W = V.T[:, : ndim]
    T = np.matmul(X, W)
    return T
