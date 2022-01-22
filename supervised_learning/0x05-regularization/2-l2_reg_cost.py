#!/usr/bin/env python3
"""
Add L2 regularization to a cost tensor.
"""

import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    """
    Calculate the lost function with tensorflow
    """
    cost = cost + tf.losses.get_regularization_loss()
    return cost
