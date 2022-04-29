#!/usr/bin/env python3
"""
GRU cell class
"""

import numpy as np


class GRUCell:
    """
    class GRUCell that represents a gated recurrent unit
    """

    def __init__(self, i, h, o):
        """
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        Creates the public instance attributes Wz, Wr, Wh, Wy, bz,
        br, bh, by that
        represent the weights and biases of the cell
            Wz and bz are for the update gate
            Wr and br are for the reset gate
            Wh and bh are for the intermediate hidden state
            Wy and by are for the output
        The weights should be initialized using a random normal
        distribution in the order listed above
        The weights will be used on the right side for matrix multiplication
        The biases should be initialized as zeros
        """
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, z):
        """
        Calculates the sigmoid activation of z
        """
        return 1 / (1 + np.exp(-z))

    def forward(self, h_prev, x_t):
        """
        x_t is a numpy.ndarray of shape (m, i) that contains the data
        input for the cell
        m is the batch size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing the previous
        hidden state
        The output of the cell should use a softmax activation function
        Returns: h_next, y
            h_next is the next hidden state
            y is the output of the cell
        """
        x = np.concatenate((h_prev, x_t), axis=1)
        z = self.sigmoid(np.matmul(x, self.Wz) + self.bz)
        r = self.sigmoid(np.matmul(x, self.Wr) + self.br)
        x = np.concatenate((r * h_prev, x_t), axis=1)
        h_hat = np.tanh(np.matmul(x, self.Wh) + self.bh)
        h_next = (1 - z) * h_prev + z * h_hat
        y = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)
        return h_next, y
