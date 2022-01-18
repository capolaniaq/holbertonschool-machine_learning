#!/usr/bin/env python3
"""
sensivity calulation
"""

import numpy as np


def sensitivity(confusion):
    """
    Calculate the sensitivity of a confusion matrix.
    """
    return np.diag(confusion) / np.sum(confusion, axis=1)
