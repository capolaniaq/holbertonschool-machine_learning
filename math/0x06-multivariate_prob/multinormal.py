#!/usr/bin/env python3
"""
Class MultivariateNormal
"""

import numpy as np


class MultiNormal:
    """Class Multinormal"""

    def __init__(self, data):
        """Constructor Class"""
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")
        self.mean = np.mean(data, axis=1).reshape(data.shape[0], 1)
        X_mean = data - self.mean
        self.cov = np.matmul(X_mean, X_mean.T) / (data.shape[1] - 1)

    def pdf(self, x):
        """
        Calculate a probability mass function
        """
        if type(x) is not np.ndarray or len(x.shape) != 2:
            raise TypeError("x must be a numpy.ndarray")
        if x.shape[1] != 1:
            raise ValueError("x must have the shape ({d}, 1)".format(self.mean.shape[0]))
        if x.shape[0] != self.mean.shape[0]:
            raise ValueError("x must have the shape ({d}, 1)".format(self.mean.shape[0]))
        det = np.linalg.det(self.cov)
        cons = 1 / (np.power((2 * np.pi), (x.shape[0] / 2)) * np.power(det, 0.5))
        inv = np.linalg.inv(self.cov)
        X_mean = x - self.mean
        exp = np.matmul(X_mean.T, np.matmul(inv, X_mean))
        pdf = float(cons * np.exp(-0.5 * exp))
        return pdf
