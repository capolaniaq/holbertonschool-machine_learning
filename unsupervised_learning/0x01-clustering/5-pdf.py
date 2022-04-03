#!/usr/bin/env python3
"""
PDF of a Gaussian Mixture Model (GMM)
"""

import numpy as np


def pdf(X, m, S):
    """
    X is a numpy.ndarray of shape (n, d) containing the data
    points whose PDF should be evaluated
    m is a numpy.ndarray of shape (d,) containing the mean of
    the distribution
    S is a numpy.ndarray of shape (d, d) containing the
    covariance of the distribution
    You are not allowed to use any loops
    You are not allowed to use the function numpy.diag or the
    method numpy.ndarray.diagonal
    Returns: P, or None on failure
    P is a numpy.ndarray of shape (n,) containing the PDF values
    for each data point
    All values in P should have a minimum value of 1e-300
    """
    # formula: 1 / ((2 * pi) ^ (d / 2) * det(S) ^ 0.5) * exp(-0.5 * (x - m) ^ T * S ^ -1 * (x - m))
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(m) is not np.ndarray or len(m.shape) != 1:
        return None
    if type(S) is not np.ndarray or len(S.shape) != 2 or S.shape[0] != S.shape[1]:
        return None
    if S.shape[0] != m.shape[0]:
        return None
    _, d = X.shape

    X_mean = X - m
    inv = np.linalg.inv(S)
    det = np.linalg.det(S)
    result = 1 / np.sqrt(((2 * np.pi) ** d) * det)

    exp = np.sum(X_mean * np.matmul(inv, X_mean.T).T, axis=1)
    pdf = result * np.exp(-0.5 * exp)
    pdf[pdf < 1e-300] = 1e-300
    return pdf