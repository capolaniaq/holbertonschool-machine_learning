#!/usr/bin/env python3
"""
Performs forward propagation for a deep RNN
"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    rnn_cells is a list of RNNCell instances of length l that will be used
    for the forward propagation
        l is the number of layers
        X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
        t is the maximum number of time steps
        m is the batch size
        i is the dimensionality of the data
        h_0 is the initial hidden state, given as a numpy.ndarray
        of shape (l, m, h)
        h is the dimensionality of the hidden state
    Returns: H, Y
    H is a numpy.ndarray containing all of the hidden states
    Y is a numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    layers = len(rnn_cells)
    _, _, h = h_0.shape
    H = np.zeros((t + 1, layers, m, h))
    H[0, :, :, :] = h_0
    Y = []

    for i in range(t):
        for j in range(layers):
            if j == 0:
                h_next, y = rnn_cells[j].forward(H[i, j, :, :], X[i, :, :])
            else:
                h_next, y = rnn_cells[j].forward(H[i, j, :, :], h_next)
            H[i + 1, j, :, :] = h_next
        Y.append(y)
    Y = np.array(Y)
    return H, Y
