#!/usr/bin/env python3
"""
Create mommentum with TensorFlow
"""

import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    Create the operation to optimize the loss using the momentum optimization
    algorithm
    :param loss: loss operation
    :param alpha: learning rate
    :param beta1: momentum weight
    :return: the operation to optimize the loss
    """
    optimizer = tf.train.MomentumOptimizer(alpha, beta1)
    return optimizer.minimize(loss)
