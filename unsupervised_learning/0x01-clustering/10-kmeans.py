#!/usr/bin/env python3
"""K-means using scikit learn"""

import sklearn.cluster


def kmeans(X, k):
    """
    performs K-means on a dataset
    """
    k_mean = sklearn.cluster.KMeans(n_clusters=k)
    k_mean.fit(X)
    clss = k_mean.labels_
    C = k_mean.cluster_centers_

    return C, clss
