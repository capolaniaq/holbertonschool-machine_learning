#!/usr/bin/env python3
"""
Class NST
"""

import tensorflow as tf
import numpy as np


class NST:
    """
    class NST
    """
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Constructor
        """
        error1 = "style_image must be a numpy.ndarray with shape (h, w, 3)"
        error2 = "content_image must be a numpy.ndarray with shape (h, w, 3)"
        if type(style_image) is not np.ndarray or len(style_image.shape) != 3 or style_image.shape[2] != 3:
            raise TypeError(error1)

        if type(content_image) is not np.ndarray or len(content_image.shape) != 3 or content_image.shape[2] != 3:
            raise TypeError(error2)

        if type(alpha) is not int and type(alpha) is not float or alpha < 0:
            raise TypeError('alpha must be a non-negative number')

        if type(beta) is not int and type(beta) is not float or beta < 0:
            raise TypeError('beta must be a non-negative number')

        tf.enable_eager_execution()
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """
        image - a numpy.ndarray of shape (h, w, 3) containing the
        image to be scaled
        if image is not a np.ndarray with the shape (h, w, 3), raise a
        TypeError with the message image must be a numpy.ndarray
        with shape (h, w, 3)
        The scaled image should be a tf.tensor with the shape
        1, h_new, w_new, 3) where max(h_new, w_new) == 512 and
        min(h_new, w_new) is scaled proportionately
        The image should be resized using bicubic interpolation
        After resizing, the image’s pixel values should be rescaled
        from the range [0, 255] to [0, 1].
        Returns: the scaled image
        """
        error = 'image must be a numpy.ndarray with shape (h, w, 3)'
        if type(image) is not np.ndarray or len(image.shape) != 3 or image.shape[2] != 3:
            raise TypeError(error)
        h, w, c = image.shape
        if h > w:
            new_h = 512
            new_w = int((new_h * w) / h)
        else:
            new_w = 512
            new_h = int((new_w * h) / w)
        image = np.expand_dims(image, axis=0)
        image = tf.image.resize_bicubic(image, (new_h, new_w))
        image = tf.cast(image, tf.float32)
        image = image / 255
        image = tf.clip_by_value(image, 0, 1)
        return image
