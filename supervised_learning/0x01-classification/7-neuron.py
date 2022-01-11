#!/usr/bin/env python3
"""
class neuron:
"""

import numpy as np
import matplotlib.pyplot as plt


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
        forward propagation function
        X is a np.array(nx, m) that contain a inputs data to the neuron
        f(x) = 1 /(1 + e^-x)
        """
        x = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-x))
        return self.__A

    def cost(self, Y, A):
        """
        Cost function
        Y that contains the correct labels for the input data
        A containing the activated output of the neuron for each example
        """
        cost = -1/A.shape[1] * np.sum((Y * np.log(A)) + ((1 - Y)*(np.log(1.0000001 - A))))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluate function
        X input for the nueron
        Y input correct for the neuron
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        A = np.where(A >= 0.5, 1, 0)
        return A, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Gradient descent function
        X input data
        Y is correct input labels
        A is the activation function output
        Alpha = learning rate
        """
        dz = A - Y
        dw = (1/Y.shape[1]) * np.matmul(X, dz.T)
        self.__W = self.__W - alpha * dw.T
        self.__b = self.__b - alpha * dz.mean()

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Train function
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        x = []
        y = []
        for i in range(iterations):
            self.__A = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
            if verbose is True or graph is True:
                if type(step) is not int:
                    raise TypeError('step must be an integer')
                if step < 0 or step > iterations:
                    raise ValueError('step must be positive and <= iterations')
            if verbose is True:
                if i == 0:
                    print("Cost after {} iterations: {}".format(i, self.cost(Y, self.__A)))
                elif i % step == 0:
                    print("Cost after {} iterations: {}".format(i, self.cost(Y, self.__A)))
                elif i == iterations:
                    print("Cost after {} iterations: {}".format(i, self.cost(Y, self.__A)))
            if graph is True:
                x.append(i)
                y.append(self.cost(Y, self.__A))
        if len(x) != 0 and len(y) != 0:
            plt.plot(x, y)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)
