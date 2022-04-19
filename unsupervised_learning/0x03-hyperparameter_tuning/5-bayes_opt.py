#!/usr/bin/env python3
"""
Class BayesianOptimization
"""

import numpy as np
from numpy import argmax
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    class BayesianOptimization
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """
        Constructor class:
        f is the black-box function to be optimized
        X_init is a numpy.ndarray of shape (t, 1) representing the inputs
        already sampled with the black-box function
        Y_init is a numpy.ndarray of shape (t, 1) representing the outputs
        of the black-box function for each input in X_init
        t is the number of initial samples
        bounds is a tuple of (min, max) representing the bounds of the space
        in which to look for the optimal point
        ac_samples is the number of samples that should be analyzed during
        acquisition
        l is the length parameter for the kernel
        sigma_f is the standard deviation given to the output of the
        black-box function
        xsi is the exploration-exploitation factor for acquisition
        minimize is a bool determining whether optimization should be
        performed for minimization (True) or maximization (False)
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        min, max = bounds
        self.X_s = np.linspace(min, max, ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculate the best sample location
        """
        mu, sig = self.gp.predict(X_s=self.X_s)

        if self.minimize:
            fx_p = np.min(self.gp.Y)
            num = fx_p - mu - self.xsi
        else:
            fx_p = np.max(self.gp.Y)
            num = mu - fx_p - self.xsi

        Z = np.where(sig == 0, 0, num / sig)
        EI = np.where(sig == 0, 0, num * norm.cdf(Z) + sig * norm.pdf(Z))
        EI = np.maximum(EI, 0)
        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI

    def optimize(self, iterations=100):
        """
        That optimizes the black-blox function
        """
        for i in range(iterations):
            X_next, _ = self.acquisition()
            if X_next in self.gp.X:
                break
            else:
                Y_next = self.f(X_next)
                self.gp.update(X_next, Y_next)
        if self.minimize:
            opt_i = np.argmin(self.gp.Y)
        else:
            opt_i = np.argmax(self.gp.Y)
        return self.gp.X[opt_i], self.gp.Y[opt_i]

