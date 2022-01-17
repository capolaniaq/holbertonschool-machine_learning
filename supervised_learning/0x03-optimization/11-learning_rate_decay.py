#!/usr/bin/env python3
"""
Updates learning rate using inverse time decay in numpy.
"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates learning rate using inverse time decay in numpy.
    alpha: initial learning rate
    decay_rate: rate at which alpha will decay
    global_step: number of passes of gradient descent that have elapsed
    decay_step: step interval
    Returns: updated learning rate
    """
    return alpha / (1 + decay_rate * (global_step // decay_step))
