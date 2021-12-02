#!/usr/bin/env python3
"""
    Task 8: Ridin' Bareback
"""


def mat_mul(mat1, mat2):
    """
        Function that multiplies two matrices
    """
    import numpy as np
    mat1 = np.array(mat1)
    mat2 = np.array(mat2)
    if mat1.shape[1] != mat2.shape[0]:
        return None
    else:
        return np.matmul(mat1, mat2).tolist()
