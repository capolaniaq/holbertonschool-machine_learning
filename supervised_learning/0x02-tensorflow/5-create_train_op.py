#!/usr/bin/env python3
"""
Calculate create_train_op.
"""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """
    Calculate train_op.
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train = optimizer.minimize(loss)
    return train
