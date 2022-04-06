#!/usr/bin/env python3
"""
Calculate the K-means algorithm with Sk-learn.
"""

import sklearn.cluster


def kmeans(X, k):
    """
    K-means algorithm.
    """
    kmeans = sklearn.cluster.KMeans(n_clusters=k)
    kmeans.fit(X)
    C = kmeans.cluster_centers_
    clss = kmeans.labels_
    return C, clss
