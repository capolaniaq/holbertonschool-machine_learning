#!/usr/bin/env python3
"""
Initialize the centroids of K-Means on the dataset X
"""

import numpy as np


def initialize(X, k):
    """
    X is a numpy.ndarray of shape (n, d) containing
    the dataset that will be used for K-means clustering
    n is the number of data points
    d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters
    Returns: a numpy.ndarray of shape (k, d) containing the
    initialized centroids for each cluster, or None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(k) is not int or k <= 0 or k > X.shape[0]:
        return None
    n, d = X.shape
    low = np.amin(X, axis=0)
    high = np.amax(X, axis=0)
    centroids = np.random.uniform(low=low, high=high, size=(k, d))
    return centroids
