#!/usr/bin/env python3
"""
Class Gaussian processes
"""

import numpy as np


class GaussianProcess:
    """
    GaussianProcess class
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Constructor class
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        Calculate the covariance matrix
        """
        sum_2 = np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + sum_2
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)
