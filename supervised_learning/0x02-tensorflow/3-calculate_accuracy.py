#!/usr/bin/env python3
"""
Calculate accuracy.
"""
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """
    Calculate ocurracy function
    """
    return tf.metrics.accuracy(y, y_pred)
