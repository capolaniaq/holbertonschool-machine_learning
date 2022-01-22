#!/usr/bin/env python3
"""
Add L2 regularization to a cost tensor.
"""

import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    """
    Calculate the lost function with tensorflow
    """
    cost_l2 = cost + tf.losses.get_regularization_losses()
    return cost_l2
