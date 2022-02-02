#!/usr/bin/env python3
"""
Forward propagation for a convolutional layer
"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
    the output of the previous layer
        m is the number of examples
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
    W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
    kernels for the convolution
        kh is the filter height
        kw is the filter width
        c_prev is the number of channels in the previous layer
        c_new is the number of channels in the output
    b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
    applied to the convolution
    activation is an activation function applied to the convolution
    padding is a string that is either same or valid, indicating the type of
    padding used
    stride is a tuple of (sh, sw) containing the strides for the convolution
        sh is the stride for the height
        sw is the stride for the width
    you may import numpy as np
    """

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev2, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = int((((h_prev - 1) * sh + kh - h_prev) / 2) + 1)
        pw = int((((w_prev - 1) * sw + kw - w_prev) / 2) + 1)
    elif padding == 'valid':
        ph = 0
        pw = 0

    new_h = int(((h_prev - kh + (2 * ph)) / sh) + 1)
    new_w = int(((w_prev - kw + (2 * pw)) / sw) + 1)

    new_image = np.zeros((m, new_h, new_w, c_new))

    for i in range(new_h):
        for j in range(new_w):
            for k in range(c_new):
                x = i * sh
                y = j * sh
                slide_image = A_prev[:, x: x + kh, y: y + kw, :]
                new_image[:, i, j, k] = np.tensordot(slide_image,
                                                     W[:, :, :, k],
                                                     axes=3) + b[:, :, :, k]

    return activation(new_image)
