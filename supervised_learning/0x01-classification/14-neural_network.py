#!/usr/bin/env python3
"""
Module to create a Neuron class
"""

import numpy as np
from numpy.core.records import format_parser


class NeuralNetwork:
    """
    class neuron
    """

    def __init__(self, nx, nodes):
        """
        Constructor
        """
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(nodes) is not int:
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        getter function for W1
        """
        return self.__W1

    @property
    def b1(self):
        """
        getter function for b1
        """
        return self.__b1

    @property
    def A1(self):
        """
        getter function for A1
        """
        return self.__A1

    @property
    def W2(self):
        """
        getter function for W2
        """
        return self.__W2

    @property
    def b2(self):
        """
        getter function for b2
        """
        return self.__b2

    @property
    def A2(self):
        """
        getter function for A2
        """
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculate a cost function
        """
        cost = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)).mean()
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron predictions
        """
        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        A2 = np.where(A2 >= 0.5, 1, 0)
        return A2, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Gradient descent for the neural network
        """
        dZ2 = A2 - Y
        dW2 = (np.matmul(dZ2, A1.T)) / A2.shape[1]
        db2 = np.sum(dZ2, axis=1, keepdims=True) / A2.shape[1]
        dZ1 = np.matmul(self.__W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = (np.matmul(dZ1, X.T)) / A1.shape[1]
        db1 = np.sum(dZ1, axis=1, keepdims=True) / A1.shape[1]
        self.__W1 = self.__W1 - (alpha * dW1)
        self.__b1 = self.__b1 - (alpha * db1)
        self.__W2 = self.__W2 - (alpha * dW2)
        self.__b2 = self.__b2 - (alpha * db2)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Method to train the neural network
        """
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')
        for i in range(iterations):
            self.__A1, self.__A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha=alpha)
        return self.evaluate(X, Y)