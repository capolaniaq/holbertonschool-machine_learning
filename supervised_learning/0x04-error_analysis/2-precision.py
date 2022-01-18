#!/usr/bin/env python3
"""
precision calulation
"""

import numpy as np


def sensitivity(confusion):
    """
    Calculate the precision of a confusion matrix.
    """
    return np.diag(confusion) / np.sum(confusion, axis=0)
