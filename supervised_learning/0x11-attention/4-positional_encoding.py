#!/usr/bin/env python3
"""
Positional Encoding
"""

import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    max_seq_len is an integer representing the maximum
        sequence length
    dm is the model depth
    Returns: a numpy.ndarray of shape (max_seq_len, dm)
        containing the positional encoding vectors
    """
    p = np.zeros((max_seq_len, dm))
    for i in range(max_seq_len):
        for j in range(dm):
            if j % 2 == 0:
                p[i, j] = np.sin(i / np.power(10000, j / dm))
            else:
                p[i, j] = np.cos(i / np.power(10000, j / dm))
    return p
