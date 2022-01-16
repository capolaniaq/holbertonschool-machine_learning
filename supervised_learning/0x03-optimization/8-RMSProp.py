#!/usr/bin/env python3
"""
RMSProp with Tensorflow
"""

import tensorflow.compat.v1 as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Creates the RMSProp optimization operation
    loss is the loss operation to minimize
    alpha is the learning rate
    beta2 is the RMSProp weight
    epsilon is a small number to avoid division by zero
    Returns: the RMSProp optimization operation
    """
    RMSprop = tf.train.RMSPropOptimizer(alpha, beta2, 0, epsilon)
    return RMSprop.minimize(loss)
