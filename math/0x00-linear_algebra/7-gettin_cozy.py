#!/us/bin/env python3

"""
    Task 5: Getting cozy
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Function that Concatenates two matrices along a specific axis
    """
    import numpy as np
    return np.concatenate((mat1, mat2), axis=axis).tolist()
