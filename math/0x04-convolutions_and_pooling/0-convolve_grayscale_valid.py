#!/usr/bin/env python3
"""
Performs a valid convolution on grayscale images
"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    images is a numpy.ndarray with shape (m, h, w) containing multiple grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw) containing the kernel for the convolution
        kh is the height of the kernel
        kw is the width of the kernel
    """
    m = images.shape[0]
    h = images.shape[1] -2
    w = images.shape[2] -2
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    images_conv = np.zeros((m, h, w))

    total_images = np.arange(m)

    for i in range(h):
        for j in range(w):
            images_conv[total_images, i, j] = (np.sum(images[total_images,
                                               i:i+kh, j:j+kw] * kernel,
                                               axis=(1, 2)))
    return images_conv
