#!/usr/bin/env python3
"""
class neuron:
"""
import numpy as np


class Neuron:
    """
    class neuron
    """

    def __init__(self, nx):
        """
        constructor class neuron
        """
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        getter function for W
        """
        return self.__W

    @property
    def b(self):
        """
        getter function for b
        """
        return self.__b

    @property
    def A(self):
        """
        getter function for A
        """
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron
        """
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A
