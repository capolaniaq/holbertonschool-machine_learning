#!/usr/bin/env python3
"""
Initialize the Gaussian Mixture Model (GMM)
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    X is a numpy.ndarray of shape (n, d) containing the data set
    k is a positive integer containing the number of clusters
    You are not allowed to use any loops
    Returns: pi, m, S, or None, None, None on failure
        pi is a numpy.ndarray of shape (k,) containing the priors
        for each cluster, initialized evenly
        m is a numpy.ndarray of shape (k, d) containing the centroid
        means for each cluster, initialized with K-means
        S is a numpy.ndarray of shape (k, d, d) containing the covariance
        matrices for each cluster, initialized as identity matrices
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(k) is not int or k <= 0:
        return None, None, None
    if k > X.shape[0]:
        return None, None, None

    C, clss = kmeans(X, k)
    pi = np.array([1 / k] * k)
    m = C
    S = np.array([np.eye(X.shape[1])] * k)
    return pi, m, S
