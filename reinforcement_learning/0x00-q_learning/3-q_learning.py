#!/usr/bin/env python3
""""
Q-learning
"""

import gym
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Function that training the Q-learning

    Args:
        env is the FrozenLakeEnv instance
        Q is a numpy.ndarray containing the Q-table
        episodes is the total number of episodes
        max_steps is the maximum number of steps per episode
        alpha is the learning rate
        gamma is the discount rate
        epsilon is the initial threshold for epsilon greedy
        min_epsilon is the minimum value that epsilon should decay
        epsilon_decay is the dacey rate for updating epsilon to be -1

    Return:
        Q is the updated Q-table
        total_rewards is the list containing the rewards per episode
    """
    total_rewards = []

    for episode in range(episodes):

        state = env.reset()
        done = False

        rewards = 0

        for s in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)

            new_state, reward, done, info = env.step(action)

            rm = (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])

            Q[state, action] = Q[state, action] + alpha * rm

            state = new_state

            if done is True:
                if reward == 0.0:
                    rewards = -1
                rewards += reward
                break

            rewards += reward
        min_ep = min_epsilon + (1 - min_epsilon)
        epsilon = min_ep * np.exp(-epsilon_decay * episode)

        total_rewards.append(rewards)

    return Q, total_rewards
