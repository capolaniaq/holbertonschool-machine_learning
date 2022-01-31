#!/usr/bin/env python3
"""
Performs a convolution on grayscale images:
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    images is a numpy.ndarray with shape (m, h, w) containing
    multiple grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        c is the number of channels in the image
    kernel is a numpy.ndarray with shape (kh, kw) containing
    the kernel for the convolution
        kh is the height of the kernel
        kw is the width of the kernel
    padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
        if ‘same’, performs a same convolution
        if ‘valid’, performs a valid convolution
        if a tuple:
            ph is the padding for the height of the image
            pw is the padding for the width of the image
    the image should be padded with 0’s
    stride is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image
    """
    m, h, w, c = images.shape
    kh, kw, c2 = kernel.shape
    sh, sw = stride

    if type(padding) is tuple:
        ph, pw = padding
    elif padding == 'same':
        ph = int((((h - 1) * sh + kh - h) / 2) + 1)
        pw = int((((w - 1) * sw + kw - w) / 2) + 1)
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        raise NameError('padding must be same, valid, or a tuple')

    nh = int(((h + (2 * ph) - kh) / sh) + 1)
    nw = int(((w + (2 * pw) - kw) / sw) + 1)

    convolved = np.zeros((m, nh, nw))

    images_p = np.pad(images, ((0, 0), (ph, ph), (pw, pw),
                      (0, 0)), 'constant', constant_values=0)

    k = kernel.copy()
    for i in range(nh):
        for j in range(nw):
            x = i * sh
            y = j * sw
            images_slide = images_p[:, x:x + kh, y:y + kw, :]
            convolved[:, i, j] = np.tensordot(images_slide, k, axes=3)
    return convolved
