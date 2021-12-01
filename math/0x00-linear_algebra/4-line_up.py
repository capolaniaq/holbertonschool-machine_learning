#!/usr/bin/env python3
"""
This module contains the function add_arrays
"""


def add_arrays(arr1, arr2):
    """
    Function that adds two arrays
    """
    import numpy as np
    if len(arr1) != len(arr2):
        return None
    else:
        return [arr1[i] + arr2[i] for i in range(len(arr1))]
