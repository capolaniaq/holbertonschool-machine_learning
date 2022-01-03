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
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        if any(list(map(lambda x: x <= 0, layers))):
            raise TypeError('layers must be a list of positive integers')
        if len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for i in range(self.L):
            key = "W{}".format(i + 1)
            if i == 0:
                self.weights["W1"] = np.random.randn(layers[i],
                                                     nx) * np.sqrt(2/nx)
            else:
                square = np.sqrt(2 / layers[i - 1])
                self.weights[key] = np.random.randn(layers[i],
                                                    layers[i - 1]) * square
            self.weights["b" + str(i + 1)] = np.zeros((layers[i], 1))
