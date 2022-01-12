#!/usr/bin/env python3
"""
Normalizes a matrix X
"""
import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization constants of a matrix X
    """
    m = np.mean(X, axis=0)
    s = np.std(X, axis=0)
    return m, s
