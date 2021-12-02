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
    if axis == 0:
        if mat1.shape[1] != mat2.shape[1]:
            return None
    elif axis == 1:
        if mat1.shape[0] != mat2.shape[0]:
            return None
    elif mat1.shape[0] == 0 or mat1.shape[1] == 0:
        return None
    
    return np.concatenate((mat1, mat2), axis=axis).tolist()
