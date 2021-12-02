#!/usr/bin/env python3
"""
    Task 13: Cats got your tongue
"""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """"
        Function that concatenates two matrices along a specific axis
    """
    matrix_concatenate = np.concatenate((mat1, mat2), axis=axis)
    return matrix_concatenate
