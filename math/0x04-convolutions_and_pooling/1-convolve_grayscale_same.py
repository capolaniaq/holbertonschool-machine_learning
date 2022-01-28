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
    m, h, w = images.shape
    kh, kw = kernel.shape
    pad_h = (kh - 1) // 2
    pad_w = (kw - 1) // 2
    images_pad = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=0)
    conv = np.zeros((m, h, w))

    total_images = np.arange(m)

    for i in range(h):
        for j in range(w):
            conv[total_images, i, j] = (np.sum(images_pad[total_images,
                                        i:i + kh, j:j + kw] * kernel,
                                        axis=(1, 2)))
    return conv

