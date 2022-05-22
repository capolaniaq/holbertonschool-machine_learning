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
    ps_ec = np.zeros((max_seq_len, dm))
    for i in range(max_seq_len):
        for j in range(dm):
            if j % 2 == 0:
                ps_ec[i, j] = np.sin(i / (10000 ** (2 * j / dm)))
            else:
                ps_ec[i, j] = np.cos(i / (10000 ** (2 * (j - 1) / dm)))
    return ps_ec
