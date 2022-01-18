#!/usr/bin/env python
"""
Create a confusion matrix for a given model.
"""

import numpy  as np


def create_confusion_matrix(labels, logits):
    """
    Create confusion matrix for a given model.

    Args:
        labels (list): list of labels
        logits (list): list of logits

    Returns:
        confusion_matrix (np.array): confusion matrix
    """
    return np.matmul(labels.T, logits)
