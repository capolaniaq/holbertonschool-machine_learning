#!/usr//bin/env python3
"""
Learning rate decay in tensorflow
"""

import tensorflow.compat.v1 as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Calculate the learning rate with tensorflow
    alpha: initial
    decay_rate: rate at which alpha will decay
    global_step: number of passes of gradient descent that have elapsed
    decay_step: step interval
    Returns: updated learning rate
    """
    alpha = tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                        decay_rate, staircase=True)
    return alpha
