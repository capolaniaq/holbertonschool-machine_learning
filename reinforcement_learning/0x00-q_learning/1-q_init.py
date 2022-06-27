#!/usr/bin/env python3
""""
Initialize Q-table
"""

import gym
import numpy as np


def q_init(env):
    """
    Function that initializes the Q-table

        Args:
            env is the FrozenLakeEnv instance

        Return:
            The Q-table as a numpy.ndarray of zeros
    """
    action_space_size = env.action_space.n
    State_space_size = env.observation_space.n

    q_table = np.zeros((State_space_size, action_space_size))

    return q_table
