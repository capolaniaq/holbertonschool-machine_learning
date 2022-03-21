#!/usr/bin/env python3
"""
Calculate the mean and covariance matrix of a data set:
"""

import numpy as np


def mean_cov(X):
    """
    X is a numpy array with shape (n, d) containing the data set:
    n is the number of data points
    d is the number of dimensions in each data point
    return mean, covariance
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")
    mean = np.mean(X, axis=(0)).reshape(1, d)
    X_mean = X - mean
    cov = np.matmul(X_mean.T, X_mean) / (n - 1)
    return mean, cov
