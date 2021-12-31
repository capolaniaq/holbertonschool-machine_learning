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
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        """
        cost = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)).mean()
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron predictions
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        new_A = np.where(A >= 0.5, 1, 0)
        return new_A, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calcules the Gradient descent
        """
        m = X.shape[1]
        dz = A - Y
        dw = (1 / m) * np.matmul(X, dz.T)
        db = (1 / m) * dz.sum()
        self.__W = self.__W - alpha * dw.T
        self.__b = self.__b - alpha * db
        return self.__W, self.__b

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neuron
        """
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if  iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')
        for i in range(iterations):
            self.__A = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
        return self.evaluate(X, Y)
