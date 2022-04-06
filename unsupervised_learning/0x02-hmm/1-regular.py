#!/usr/bin/env python3
"""
Calculate the Steady State Probability of a Markov Chain
"""

import numpy as np


def regular(P):
    """
    P is a is a square 2D numpy.ndarray of shape (n, n) representing the
    transition matrix
    P[i, j] is the probability of transitioning from state i to state j
    n is the number of states in the markov chain
    Returns: a numpy.ndarray of shape (1, n) containing the steady state
    probabilities, or None on failure
    """
    if type(P) is not np.ndarray or P.ndim != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None

    if np.all(P * P) == 0:
        return None

    eigen_val, eigen_vec = np.linalg.eig(P.T)
    close_1 = np.isclose(eigen_val, 1)
    target_v = eigen_vec[:, close_1]
    target_v = target_v[:, 0]
    return target_v / np.sum(target_v)
