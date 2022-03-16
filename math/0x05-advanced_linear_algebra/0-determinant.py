#!/usr/bin/env python3
"""
Module that calculate a determinant form matrix
"""

import numpy as np


def determinant(matrix):
    """
    Function that calculate a determinant
    """
    if type(matrix) is not list:
        raise TypeError("matrix must be a list of lists")
    for i in matrix:
        if len(matrix) == 1 and len(i) == 0:
            return 1
        if type(i) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(matrix) != len(i):
            raise ValueError("matrix must be a square matrix")
    return int(np.linalg.det(matrix))
