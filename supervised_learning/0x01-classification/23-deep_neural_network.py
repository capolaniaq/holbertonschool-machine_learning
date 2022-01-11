#!/usr/bin/env python3
"""
Module DeepNeuralNetwork
"""

import numpy as np
import matplotlib.pyplot as plt


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

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        X is a numpy.ndarray with shape (nx, m)
        nx is the number of input features to the neuron
        m is the number of examples
        """
        self.__cache['A0'] = X
        for i in range(self.__L):
            bais = self.__weights['b' + str(i + 1)]
            input = self.__cache['A' + str(i)]
            weight = self.__weights['W' + str(i + 1)]
            key = 'A{}'.format(i + 1)
            z = np.matmul(weight, input) + bais
            self.__cache[key] = 1 / (1 + np.exp(-z))
        return self.__cache["A" + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """
        Cost function logistic regresion
        Y = correct output
        A = output for the Activation function
        """
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluate function
        X input data
        Y correct output
        """
        A, prediction = self.forward_prop(X)
        cost = self.cost(Y, A)
        A = np.where(A >= 0.5, 1, 0)
        return A, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Gradient descent function
        Y = correct output function
        cache = dicctionary with A's
        alpha = learning rate
        """
        m = Y.shape[1]
        weights = self.__weights.copy()
        for i in range(self.__L, 0, -1):
            A = cache['A' + str(i)]
            if i == self.__L:
                dz = A - Y
            else:
                dz = np.matmul(weights['W' + str(i + 1)].T, dz) * (A * (1 - A))
            dw = 1 / m * np.matmul(dz, cache['A' + str(i - 1)].T)
            db = 1 / m * np.sum(dz, axis=1, keepdims=True)
            self.__weights["W" + str(i)] = weights['W' + str(i)] - alpha * dw
            self.__weights["b" + str(i)] = weights['b' + str(i)] - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        train function
        iterations is number of updates from cache and weights
        alpha is a learning rate
        X is the input
        Y is the correct output
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
            last_cache, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            if verbose is True or graph is True:
                if type(step) is not int:
                    raise TypeError('step must be an integer')
                if step < 0 or step > iterations:
                    raise ValueError('step must be positive and <= iterations')
            if verbose is True:
                if i == 0:
                    print("Cost after {} iterations: {}".format(i, self.cost(Y, last_cache)))
                elif i % step == 0:
                    print("Cost after {} iterations: {}".format(i, self.cost(Y, last_cache)))
                elif i == iterations:
                    print("Cost after {} iterations: {}".format(i, self.cost(Y, last_cache)))
            if graph is True:
                x.append(i)
                y.append(self.cost(Y, last_cache))
        if len(x) != 0 and len(y) != 0:
            plt.plot(x, y)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)
