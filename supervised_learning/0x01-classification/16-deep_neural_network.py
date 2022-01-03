#!/usr/bin/env python3
"""
Module DeepNeuralNetwork
"""

import numpy as np


class DeepNeuralNetwork:
    """
    Class DeepNeuralNetwork
    """

    def __init__(self, nx, layers):
        """
        Construcctor
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if len(layers) == 0 or type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for i in range(self.L):
            if type(layers[i]) is not int or layers[i] < 0:
                raise TypeError("layers must be a list of positive integers")
            key = "W{}".format(i + 1)
            self.weights[key] = np.random.randn(layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
            key = "b{}".format(i + 1)
            self.weights[key] = np.zeros((layers[i - 1], layers[i]))
