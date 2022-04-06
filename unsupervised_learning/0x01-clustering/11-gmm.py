#!/usr/bin/env python3
"""
    11-gmm.py - Gaussian Mixture Model
"""

import sklearn.mixture


def gmm(X, k):
    """
    GMM with sklearn
    """
    gmm = sklearn.mixture.GaussianMixture(n_components=k)
    gmm.fit(X)
    pi = gmm.weights_
    m = gmm.means_
    S = gmm.covariances_
    clss = gmm.predict(X)
    bic = gmm.bic(X)
    return pi, m, S, clss, bic
