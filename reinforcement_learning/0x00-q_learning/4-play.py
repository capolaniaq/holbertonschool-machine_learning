#!/usr/bin/env python3
""""
Play
"""

import gym
import numpy as np


def play(env, Q, max_steps=100):
    """
    Function that has trained agend play an epsode

    Args:
        env is the FrozenLakeEnv instance
        Q is a numpy.ndarray containing the q-table
        max_steps is the maximum number of steps in the episode

    Return:
        The total rewards fro epidose
    """
    state = env.reset()
    env.render()
    done = False
    rewards = 0

    for step in range(max_steps):
        action = np.argmax(Q[state, :])

        new_state, reward, done, info = env.step(action)

        rewards += reward
        env.render()
        state = new_state

        if done is True:
            break

    env.close()

    return rewards
