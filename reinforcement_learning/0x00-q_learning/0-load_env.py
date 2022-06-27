#!/usr/bin/env python3
""""
Load the Environment
"""

import gym
import numpy as np


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Function that loads the pre-made FrozenLakeEnv

    Args:
        desc: either None or a list of list containing custom descripcion
              of the map to load for the environment
        map_name: either None or a string containing the pre-made map model
        is_slippery: es a booena to determine if the ice is slippery

    Return:
        the environment
    """
    env = gym.make('FrozenLake-v0', desc=desc, map_name=map_name,
                   is_slippery=is_slippery)
    return env
