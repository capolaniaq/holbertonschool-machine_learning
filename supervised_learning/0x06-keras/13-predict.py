#!/usr/bin/env python3
"""
Functions to make a prediction with a model
"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    network is the network model to make the prediction with
    data is the input data to make the prediction with
    verbose is a boolean that determines if output should be
    printed during the prediction process
    Returns: the prediction for the data
    """
    return network.predict(data, verbose=verbose)
