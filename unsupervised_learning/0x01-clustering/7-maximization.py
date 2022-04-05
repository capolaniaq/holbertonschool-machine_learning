#!/usr/bin/env python3
"""
Maximization step of the EM algorithm.
"""

import numpy as np


def maximization(X, g):
    """
    X is a numpy.ndarray of shape (n, d) containing the data set
    g is a numpy.ndarray of shape (k, n) containing the posterior probabilities
    for each data point in each cluster
    You may use at most 1 loop
    Returns: pi, m, S, or None, None, None on failure
    pi is a numpy.ndarray of shape (k,) containing the updated priors for each
    cluster
    m is a numpy.ndarray of shape (k, d) containing the updated centroid means
    for each cluster
    S is a numpy.ndarray of shape (k, d, d) containing the updated covariance
    matrices for each cluster
    """
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None, None
    if type(g) is not np.ndarray or g.ndim != 2:
        return None, None, None
    n, d = X.shape
    k, _ = g.shape
    if g.shape[1] != n:
        return None, None, None
    if np.isclose([np.sum(g)], [1])[0]:
        return None, None, None

    S = np.zeros((k, d, d))
    pi = np.zeros((k,))
    m = np.matmul(g, X) / np.sum(g, axis=1).reshape(-1, 1)
    for i in range(k):
        X_mean = X - m[i]
        S[i] = np.matmul(np.multiply(g[i], X_mean.T), X_mean) / np.sum(g[i])
        pi[i] = np.sum(g[i]) / n
    return pi, m, S
