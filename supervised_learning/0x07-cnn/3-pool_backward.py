#!/usr/bin/env python3
"""
Backpropagation in pooling layer for Convolutional Neural Networks
"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    dA is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
    partial derivatives with respect to the output of the pooling layer
        m is the number of examples
        h_new is the height of the output
        w_new is the width of the output
        c is the number of channels
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c) containing
    the output of the previous layer
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
    kernel_shape is a tuple of (kh, kw) containing the size of the
    kernel for the pooling
        kh is the kernel height
        kw is the kernel width
    stride is a tuple of (sh, sw) containing the strides for the pooling
        sh is the stride for the height
        sw is the stride for the width
    mode is a string containing either max or avg, indicating whether to
    perform maximum or average pooling,
    """
    m, h_new, w_new, c = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        for j in range(h_new):
            for k in range(w_new):
                for ch in range(c):
                    x = j * sh
                    y = k * sw
                    x_end = x + kh
                    y_end = y + kw
                    if mode == 'avg':
                        dA_prev[i, x:x_end, y:y_end, ch] += dA[i,
                                                               j, k,
                                                               ch] / (kh * kw)
                    elif mode == 'max':
                        a_prev_slice = A_prev[i, x:x_end, y:y_end, ch]
                        mask = (a_prev_slice == np.max(a_prev_slice))
                        dA_prev[i, x:x_end, y:y_end, ch] += dA[i, j,
                                                               k, ch] * mask

    return dA_prev
