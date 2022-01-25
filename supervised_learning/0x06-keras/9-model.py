#!/usr/bin/env python3
"""
Funtions to save and load neural network models
"""

import tensorflow.keras as K


def save_model(network, filename):
    """
    saves a model to a file
    """
    network.save(filename)


def load_model(filename):
    """
    loads a model from a file
    """
    return K.models.load_model(filename)
