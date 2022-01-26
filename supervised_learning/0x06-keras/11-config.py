#!/usr/bin/env python3
"""
Functions to save and load the configuration of a model
"""

import tensorflow.keras as K


def save_config(network, filename):
    """
    saves a models configuration in JSON format:
    network is the model whose configuration should be saved
    filename is the path of the file that the configuration
    should be saved to
    Returns: None
    """
    config = network.to_json()
    with open(filename, 'w') as file:
        file.write(config)

def load_config(filename):
    """
    loads a model with a specific configuration:
    filename is the path of the file containing the models
    configuration in JSON format
    Returns: the loaded model
    """
    with open(filename) as js_config:
        config = js_config.read()
    return K.models.model_from_json(config)
