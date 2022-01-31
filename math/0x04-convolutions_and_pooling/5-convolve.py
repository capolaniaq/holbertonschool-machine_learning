#!/usr/bin/env python3
"""
Performs a convolution on grayscale images:
"""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    m, h, w, c = images.shape
    kh, kw, c2, cn = kernels.shape
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

    convolved = np.zeros((m, nh, nw, cn))

    images_p = np.pad(images, ((0, 0), (ph, ph),
                               (pw, pw), (0, 0)), 'constant')

    ker = kernels.copy()
    for i in range(nh):
        for j in range(nw):
            for k in range(cn):
                x = i * sh
                y = j * sw
                images_slide = images_p[:, x:x + kh, y:y + kw, :]
                convolved[:, i, j, k] = np.tensordot(images_slide,
                                                     ker[:, :, :, k], axes=3)
    return convolved
