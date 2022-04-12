#!/usr/bin/env python3
"""
backward algorithm for the HMM
"""

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Observation is a numpy.ndarray of shape (T,) that contains the index
    of the observation
        T is the number of observations
    Emission is a numpy.ndarray of shape (N, M) containing the emission
    probability of a specific observation given a hidden state
    Emission[i, j] is the probability of observing j given the hidden state i
        N is the number of hidden states
        M is the number of all possible observations
    Transition is a 2D numpy.ndarray of shape (N, N) containing the
    transition probabilities
    Transition[i, j] is the probability of transitioning from
    the hidden state i to j
    Initial a numpy.ndarray of shape (N, 1) containing the probability of
    starting in a particular hidden state
    Returns: P, B, or None, None on failure
        Pis the likelihood of the observations given the model
        B is a numpy.ndarray of shape (N, T) containing the backward
        path probabilities
        B[i, j] is the probability of generating the future observations
        from hidden state i at time j8
    """
    try:
        T = Observation.shape[0]
        N = Initial.shape[0]

        beta = np.zeros((N, T))

        beta[:, T - 1] = np.ones((N))
        for t in range(T - 2, -1, -1):
            for j in range(N):
                tr = Transition[j, :]
                em = Emission[:, Observation[t + 1]]
                beta[j, t] =  np.sum(beta[:, t + 1] * tr * em)

        P = np.sum(beta[:, 0] * Initial * Emission[:, Observation[0]])
        return P, beta

    except Exception:
        return None, None