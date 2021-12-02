#!/usr/bin/env python3
"""
    Task 5:
"""


def add_matrices2D(mat1, mat2):
    """
        Adds two matrices
    """
    import numpy as np
    mat1 = np.matrix(mat1)
    mat2 = np.matrix(mat2)
    if mat1.shape != mat2.shape:
        return None
    add_matrix = mat1 + mat2
    return add_matrix.tolist()
