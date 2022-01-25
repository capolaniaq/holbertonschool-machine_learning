#!/usr/bin/env python3
"""
converts a label vector into a one-hot matrix
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    The last dimension of the one-hot matrix must be the number of classes
    Returns: the one-hot matrix
    """
    one_hot_matrix = K.utils.to_categorical(labels, num_classes=classes)
    return one_hot_matrix
