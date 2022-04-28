#!/usr/bin/env python3
"""
Create the class RNNCell that represents a cell of a simple RNN:
"""

import numpy as np


class RNNCell:
    """RNNCell class"""

    def __init__(self, i, h, o):
        """
        Constructor class
            i is the dimensionality of the data
            h is the dimensionality of the hidden state
            o is the dimensionality of the outputs
            Creates the public instance attributes Wh, Wy,
                bh, by that represent the weights and biases
                of the cell
                Wh and bh are for the concatenated hidden state
                and input data
                Wy and by are for the output
            The weights should be initialized using a random
            normal distribution in the order listed above
            The weights will be used on the right side for
            matrix multiplication
            The biases should be initialized as zeros
        """
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        performs forward propagation for one time step
        x_t is a numpy.ndarray of shape (m, i) that contains
        the data input for the cell
            m is the batch size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing
        the previous hidden state
        The output of the cell should use a softmax activation
        function
        Returns: h_next, y
            h_next is the next hidden state
            y is the output of the cell
        """
        whh = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(whh, self.Wh) + self.bh)
        y = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)
        return h_next, y
