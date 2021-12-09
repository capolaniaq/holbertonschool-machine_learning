#!/usr/bin/env python3
"""
This module contains the Summation squared
"""


def summation_i_squared(n):
    """
    Returns the summation of the squared integers
    """
    squared = n*n
    if n == 1:
        return squared
    else:
        return summation_i_squared(n-1) + squared
