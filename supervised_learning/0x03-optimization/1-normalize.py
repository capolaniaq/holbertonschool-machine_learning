#!/usr/bin/env python3
"""
Normalizes a matrix X
"""
import numpy as np


def normalize(X, m, s):
    """
    Normalizes a matrix X
    """
    return (X - m) / s
