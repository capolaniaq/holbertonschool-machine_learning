#!/usr/bin/env python3
"""
Convolutional Neural Network with Pooling
"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
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