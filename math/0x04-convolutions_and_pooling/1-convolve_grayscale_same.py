#!/usr/bin/env python3
"""
Performs a same convolution on grayscale images
"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    images is a numpy.ndarray with shape (m, h, w) containing multiple grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw) containing the kernel for the convolution
        kh is the height of the kernel
        kw is the width of the kernel
    if necessary, the image should be padded with 0’s
    """
    images = np.pad(images, ((0,0), (1, 1), (1, 1)), 'constant', constant_values=0)
    m, h, w = images.shape
    kh, kw = kernel.shape
    new_h = h - kh + 1
    new_w = w - kw + 1
    conv = np.zeros((m, new_h, new_w))

    for i in range(new_h):
        for j in range(new_w):
            conv[:, i, j] = np.sum(images[:,
                                   i:i+kh, j:j+kw] * kernel,
                                   axis=(1, 2))
    return conv
