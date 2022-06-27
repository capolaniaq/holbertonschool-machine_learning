#!/usr/bin/env python3
""""
Epsilon Greedy
"""

import gym
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    That uses epsilon-greedy to determine the next section

    Args:
        Q is the numpy.ndarray containing the q-table
        state is the current state
        epsilon is the epsilon to use for the calculation

    Return:
        The next action index
    """
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.randint(Q.shape[1])
    else:
        action = np.argmax(Q[state, :])

    return action
