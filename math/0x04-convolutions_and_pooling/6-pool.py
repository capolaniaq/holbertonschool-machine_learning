#!/usr/bin/env python3
"""
Convolutional Neural Network with Pooling
"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    images is a numpy.ndarray with shape (m, h, w, c)
    containing multiple images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        c is the number of channels in the image
    kernel_shape is a tuple of (kh, kw) containing the
    kernel shape for the pooling
        kh is the height of the kernel
        kw is the width of the kernel
    stride is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image
    mode indicates the type of pooling
    max indicates max pooling
    avg indicates average pooling
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    new_h = int(((h - kh) / sh) + 1)
    new_w = int(((w - kw) / sw) + 1)

    convol = np.zeros((m, new_h, new_w, c))

    for i in range(new_h):
        for j in range(new_w):
            x = i * sh
            y = j * sw
            if mode == 'max':
                image_slide = images[:, x:x + kh, y:y + kw, :]
                convol[:, i, j, :] = np.max(image_slide, axis=(1, 2))
            elif mode == 'avg':
                image_slide = images[:, x:x + kh, y:y + kw, :]
                convol[:, i, j, :] = np.mean(image_slide, axis=(1, 2))
    return convol
