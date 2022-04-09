#!/usr/bin/env python3
"""
Absorbing Markov chains
"""

import numpy as np


def get_to_abs_state(keys, i, P):
    """
    keys is a list of the absorbing states
    i is the current state
    P is the transition matrix
    Returns: True if it is absorbing, or False on failur
    """
    x = P.T[i]
    for j in range(P.shape[0]):
        if x[j] > 0:
            keys.append(j)
    return keys


def absorbing(P):
    """
    P is a is a square 2D numpy.ndarray of shape (n, n)
    representing the standard transition matrix
    P[i, j] is the probability of transitioning from state i to state j
    n is the number of states in the markov chain
    Returns: True if it is absorbing, or False on failur
    """
    if type(P) is not np.ndarray or P.ndim != 2:
        return None
    if np.any(np.sum(P, axis=1) != 1):
        return None
    x = np.diagonal(P)

    if not np.any(x == 1):
        return False

    if np.all(x == 1):
        return True

    keys = []

    for i in range(len(x)):
        if x[i] == 1:
            keys.append(i)

    for i in range(P.shape[0]):
        if i in keys:
            keys = get_to_abs_state(keys, i, P)

    for i in range(P.shape[0]):
        if i in keys:
            keys = get_to_abs_state(keys, i, P)

    return len(set(keys)) == P.shape[0]
