#!/usr/bin/env python3
"""
Module DeepNeuralNetwork
"""

import numpy as np


class DeepNeuralNetwork:
    """
    Deep Neural Network Class
    """

    def __init__(self, nx, layers):
        """
        Constructor
        Exceptions for the use and operative the class
        L =  indicated the numbers of layers in the neural network
        cache = is a dictionary, to hold intermediary values of the network
        weights = a dictionary to hold the all wights and baised of the
        neural network
        """
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) is not list or len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')
        if len(list(filter(lambda x: x < 0, layers))) != 0:
            raise TypeError('layers must be a list of positive integers')
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(self.L):
            key = "W{}".format(i + 1)
            if i == 0:
                self.weights[key] = np.random.randn(layers[i],
                                                    nx) * np.sqrt(2/nx)
            else:
                square = np.sqrt(2 / layers[i - 1])
                self.__weights[key] = np.random.randn(layers[i],
                                                      layers[i - 1]) * square
            self.__weights['b' + str(i + 1)] = np.zeros(shape=(layers[i], 1))

    @property
    def L(self):
        """
        Getter for L atribute
        """
        return self.__L

    @property
    def cache(self):
        """
        Getter for cache atribute
        """
        return self.__cache

    @property
    def weights(self):
        """
        getter for the weights
        """
        return self.__weights
