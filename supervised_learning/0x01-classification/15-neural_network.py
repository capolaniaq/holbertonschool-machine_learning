#!/usr/bin/env python3
"""
Module to create a Neuron class
"""

import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """
    Neural Network class
    """
    def __init__(self, nx, nodes):
        """
        constructor
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros(shape=(3, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """
        Forward propagtion function
        use the sigmoid function for the obtain data
        X contain de input data
        X.shape = (nx. m)
        nx = is the number of the input features
        m = is the number of the examples
        update __A1 and __A2
        return __A1 and __A2
        __A1 = actived output for the hidden layer
        __A2 = actived output for the neuron(prediction)
        """
        z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))
        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Cost function
        Y is the correct output
        A is the Actived output(prediction)
        return the float to correspont the cost
        """
        cost = -(1/Y.shape[1]) * (np.sum((Y*np.log(A)) + (1 - Y)* np.log(1.0000001 - A)))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluate function
        X input for the neural network
        Y correct input data
        return prediction and cost respectively
        """
        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        A2 = np.where(A2 >= 0.5, 1, 0)
        return A2, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Gradient descent for the neural network
        X input
        Y correct input
        A1 activation output for the hidden layers
        A2 activation output for the neuron
        Alpha is the reason to the change or move
        update the __W1, __b1, __W2, __b2
        dz = A - Y
        dw = (1 / Y.shape[1]) * np.matmul(X, dz.T)
        self.__W = self.__W - alpha * dw.T
        self.__b = self.__b - alpha * dz.mean()
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

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Function to train the neural network
        """
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')
        x = []
        y = []
        for i in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)
            if verbose is True or graph is True:
                if type(step) is not int:
                    raise TypeError('step must be an integer')
                if step < 0 or step > iterations:
                    raise ValueError('step must be positive and <= iterations')
            if verbose is True:
                if i == 0:
                    print("Cost after {} iterations: {}".format(i, self.cost(Y, self.__A2)))
                elif i % step == 0:
                    print("Cost after {} iterations: {}".format(i, self.cost(Y, self.__A2)))
                elif i == iterations:
                    print("Cost after {} iterations: {}".format(i, self.cost(Y, self.__A2)))
            if graph is True:
                x.append(i)
                y.append(self.cost(Y, self.__A2))
        if len(x) != 0 and len(y) != 0:
            plt.plot(x, y)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)
