#!/usr/bin/env python3
"""
    Task 7: Getting cozy
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Function that Concatenates two matrices along a specific axis
    """
    import numpy as np
    mat1 = np.array(mat1)
    mat2 = np.array(mat2)
    if mat1.shape != mat2.shape:
        return None
    return np.concatenate((mat1, mat2), axis=axis).tolist()
