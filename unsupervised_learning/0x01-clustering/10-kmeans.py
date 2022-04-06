#!/usr/bin/env python3
"""
Calculate the K-means algorithm with Sk-learn.
"""

import numpy as np
import skelarn.cluster


def kmeans(X, k):
    """
    X is a numpy.ndarray of shape (n, d) containing the dataset
    k is the number of clusters
    The only import you are allowed to use is import sklearn.cluster
    Returns: C, clss
        C is a numpy.ndarray of shape (k, d) containing the centroid
        means for each cluster
        clss is a numpy.ndarray of shape (n,) containing the index of
        the cluster in C that each data point belongs to
    """
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None
    if type(k) is not int or k < 1:
        return None, None
    kmeans = skelarn.cluster.KMeans(n_clusters=k).fit(X)
    C = kmeans.cluster_centers_
    clss = kmeans.labels_
    return C, clss
