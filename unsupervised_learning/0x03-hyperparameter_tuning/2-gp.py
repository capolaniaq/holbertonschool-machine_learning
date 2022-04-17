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
        mw = 2 * np.dot(X1, X2.T)
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - mw
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        """
        Function that predicts the mean and standard deviation
        Returns mu and sigma
        """
        k = self.kernel(self.X, self.X)
        k_s = self.kernel(self.X, X_s)
        k_ss = self.kernel(X_s, X_s)
        k_inv = np.linalg.inv(k)

        mu = np.dot(np.dot(k_s.T, k_inv), self.Y)
        sigma = k_ss - np.dot(np.dot(k_s.T, k_inv), k_s)
        return mu.reshape(-1, ), np.diag(sigma)

    def update(self, X_new, Y_new):
        """
        Updates the Gaussian processes
        """
        self.X = np.concatenate((self.X, X_new.reshape(-1, 1)), axis=0)
        self.Y = np.concatenate((self.Y, Y_new.reshape(-1, 1)), axis=0)
        self.K = self.kernel(self.X, self.X)
