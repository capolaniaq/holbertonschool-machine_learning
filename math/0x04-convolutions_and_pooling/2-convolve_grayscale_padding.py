#!/usr/bin/env python3
"""
Performs a convolution on grayscale images with custom padding:
"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    images is a numpy.ndarray with shape (m, h, w) containing
    multiple grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw) containing
    the kernel for the convolution
        kh is the height of the kernel
        kw is the width of the kernel
    padding is a tuple of (ph, pw)
        ph is the padding for the height of the image
        pw is the padding for the width of the image
    the image should be padded with 0’s
    """
    ph, pw = padding
    images_pad = np.pad(images, ((0, 0), (ph, ph),
                                 (pw, pw)), 'constant')

    m, h, w = images_pad.shape
    kh, kw = kernel.shape

    new_h = h - kh + 1
    new_w = w - kw + 1

    conv = np.zeros((m, new_h, new_w))

    total_images = np.arange(m)

    for i in range(h):
        for j in range(w):
            conv[total_images, i, j] = (np.sum(images_pad[total_images,
                                        i:i + kh, j:j + kw] * kernel,
                                        axis=(1, 2)))
    return conv
