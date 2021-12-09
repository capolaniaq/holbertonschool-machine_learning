#!/usr/bin/env python3
"""
This module contains the Summation squared
"""


def summation_i_squared(n):
    """
    Returns the summation of the squared integers
    """
    import numpy as np
    if type(n) is not int or n < 1:
        return None
    sqaured = np.arange(1, n + 1)
    summ = np.sum(sqaured ** 2)
    return summ
