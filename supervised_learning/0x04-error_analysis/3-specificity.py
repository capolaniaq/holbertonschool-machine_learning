#!/usr/bin/env python3
"""
Calculate the specificity of a confusion matrix.
"""

import numpy as np


def specificity(confusion):
    """
    calculates the specificity for each class in a confusion matrix:
    """
    TP = np.diag(confusion)
    FP = np.sum(confusion, axis=0) - TP
    FN = np.sum(confusion, axis=1) - TP
    TN = np.sum(confusion) - (TP + FN + FP)
    return TN / (TN + FP)
