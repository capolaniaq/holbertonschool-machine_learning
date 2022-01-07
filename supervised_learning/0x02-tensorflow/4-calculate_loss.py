#!/usr/bin/env python3
"""
Calculate loss
"""
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """
    Calculate loss function
    """
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    return loss
