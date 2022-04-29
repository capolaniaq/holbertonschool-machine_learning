#!/usr/bin/env python3
"""
:
"""

import numpy as np


class BidirectionalCell:
    """
    class BidirectionalCell that Represents a bidirectional cell of an RNN
    """

    def __init__(self, i, h, o):
        """
        i is the dimensionality of the data
        h is the dimensionality of the hidden states
        o is the dimensionality of the outputs
        Creates the public instance attributes Whf, Whb, Wy, bhf, bhb,
        by that represent the weights and biases of the cell
        Whf and bhfare for the hidden states in the forward direction
        Whb and bhbare for the hidden states in the backward direction
        Wy and byare for the outputs
        The weights should be initialized using a random normal distribution
        in the order listed above
        The weights will be used on the right side for matrix multiplication
        The biases should be initialized as zeros
        """
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(2 * h, o)
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        x_t is a numpy.ndarray of shape (m, i) that contains the data input
        for the cell
        m is the batch size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing the previous
        hidden state
        Returns: h_next, the next hidden state
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concat, self.Whf) + self.bhf)
        return h_next
