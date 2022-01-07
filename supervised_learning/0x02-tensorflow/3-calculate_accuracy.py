#!/usr/bin/env python3
"""
Calculate accuracy.
"""
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """
    Calculate ocurracy function
    """
    ocurracy = tf.reduce_mean(tf.square(y / y_pred))
    return ocurracy
