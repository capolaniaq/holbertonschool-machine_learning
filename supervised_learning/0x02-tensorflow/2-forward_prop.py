#!/usr/bin/env python3
"""
Create a first layer of a new project.
"""
import tensorflow.compat.v1 as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Create a forward propagation function.
    """
    prev = x
    for i in range(len(layer_sizes)):
        prev = create_layer(prev, layer_sizes[i], activations[i])
    return prev
