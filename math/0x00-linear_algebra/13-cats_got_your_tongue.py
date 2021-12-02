#!/usr/bin/env python3
"""
    Task 13: Cats got your tongue
"""


def np_cat(mat1, mat2, axis=0):
    """"
        Function that concatenates two matrices along a specific axis
    """
    import numpy as np
    matrix_concatenate = np.concatenate((mat1, mat2), axis=axis)
    return matrix_concatenate