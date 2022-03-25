#!/usr/bin/env python3
"""
Calculate the likelihood from 1D binomial distribution
"""

import numpy as np


def likelihood(x, n, P):
    """
    x is the number of patients that develop severe side effects
    n is the total number of patients observed
    P is a 1D numpy.ndarray containing the various hypothetical
    probabilities of developing severe side effects
    Returns: a 1D numpy.ndarray containing the likelihood of
    obtaining the data, x and n, for each probability in P, respectively
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")

    if type(x) is not int or x < 0:
        raise ValueError("x must be an integer\
                         that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if [x for x in P if x < 0 or x > 1]:
        raise ValueError("All values in P must be in the range [0, 1]")

    P_p = np.zeros(P.shape)
    n_f = np.math.factorial(n)
    x_f = np.math.factorial(x)
    p_f = np.math.factorial(n - x)
    for i in range(len(P)):
        P_p[i] = (n_f / (p_f * x_f)) * (P[i] ** x) * ((1 - P[i]) ** (n - x))
    return P_p
