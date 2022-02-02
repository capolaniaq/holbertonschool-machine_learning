#!/usr/bin/env python3
"""
Forward propagation over a pooling layer of a neural network
"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
        m is the number of examples
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
    kernel_shape is a tuple of (kh, kw) containing the size of the kernel
    for the pooling
        kh is the kernel height
        kw is the kernel width
    stride is a tuple of (sh, sw) containing the strides for the pooling
        sh is the stride for the height
        sw is the stride for the width
    mode is a string containing either max or avg, indicating whether to
    perform maximum or average pooling, respectively
    """

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    new_h = int(((h_prev - kh) / sh) + 1)
    new_w = int(((w_prev - kw) / sw) + 1)

    pool_convol = np.zeros((m, new_h, new_w, c_prev))

    for i in range(new_h):
        for j in range(new_w):
            for k in range(c_prev):
                x = i * sh
                y = j * sw
                if mode == 'max':
                    image_slide = A_prev[:, x:x + kh, y:y + kw, :]
                    pool_convol[:, i, j, :] = np.max(image_slide, axis=(1, 2))
                elif mode == 'avg':
                    image_slide = A_prev[:, x:x + kh, y:y + kw, :]
                    pool_convol[:, i, j, :] = np.mean(image_slide, axis=(1, 2))
    return pool_convol
