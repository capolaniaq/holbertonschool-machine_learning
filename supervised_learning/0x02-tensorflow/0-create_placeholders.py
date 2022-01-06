#!/usr/bin/env python3
"""
Place Holder module
"""
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """
    Function that creates the placeholders needed for the model:
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return x, y
