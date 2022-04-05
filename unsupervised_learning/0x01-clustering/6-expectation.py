#!/usr/bin/env python3
"""
Expectation step of the EM algorithm for Gaussian Mixture Model (GMM).
"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    X is a numpy.ndarray of shape (n, d) containing the data set
    pi is a numpy.ndarray of shape (k,) containing the priors for each
    cluster
    m is a numpy.ndarray of shape (k, d) containing the centroid means
    for each cluster
    S is a numpy.ndarray of shape (k, d, d) containing the covariance
    matrices for each cluster
    You may use at most 1 loop
    Returns: g, l, or None, None on failure
    g is a numpy.ndarray of shape (k, n) containing the posterior
    probabilities
    for each data point in each cluster
    l is the total log likelihood
    """
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None
    if type(pi) is not np.ndarray or pi.ndim != 1:
        return None, None
    if type(m) is not np.ndarray or m.ndim != 2:
        return None, None
    if type(S) is not np.ndarray or S.ndim != 3:
        return None, None
    n, d = X.shape
    k = pi.shape[0]
    if m.shape[0] != k or m.shape[1] != d:
        return None, None
    if S.shape[0] != k or S.shape[1] != d or S.shape[2] != d:
        return None, None
    post = np.zeros((k, n))
    for i in range(k):
        post[i] = pi[i] * pdf(X, m[i], S[i])
    marginal = np.sum(post, axis=0)
    likelihood = np.sum(np.log(marginal))
    return post / marginal, likelihood
