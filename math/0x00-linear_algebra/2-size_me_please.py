#!/usr/bin/env python3
"""
    task 3: Shape of a matrix
"""
import numpy as np

def matrix_shape(matrix):
    """
    Returns the shape of a matrix
    """
    new_matrix = np.array(matrix)
    return list(new_matrix.shape)
