#!/usr/bin/env python3
"""Neural Style Tranfer NST Class"""

import numpy as np
import tensorflow as tf


class NST:
    """performs tasks for neural style transfer
    Class atributes:
        - Content layer where will pull our feature maps
        - Style layer we are interested in
    """
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """ Class constructor
        Arg:
            - style_image: img used as a style reference, numpy.ndarray
            - content_image: image used as a content reference, numpy.ndarray
            - alpha: the weight for content cost
            - beta: the weight for style cost
        Environment:
            Eager execution: TensorFlow’s imperative programming
                             environment, evaluates operations immediately
        """
        sty_error = 'style_image must be a numpy.ndarray with shape (h, w, 3)'
        c_error = "content_image must be a numpy.ndarray with shape (h, w, 3)"
        if not isinstance(style_image, np.ndarray):
            raise TypeError(sty_error)
        if len(style_image.shape) != 3:
            raise TypeError(sty_error)
        if style_image.shape[2] != 3:
            raise TypeError(sty_error)

        if not isinstance(content_image, np.ndarray):
            raise TypeError(c_error)
        if len(content_image.shape) != 3:
            raise TypeError(c_error)
        if content_image.shape[2] != 3:
            raise TypeError(c_error)

        if isinstance(alpha, str):
            raise TypeError("alpha must be a non-negative number")
        if alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if isinstance(beta, str):
            raise TypeError("beta must be a non-negative number")
        if beta < 0:
            raise TypeError("beta must be a non-negative number")

        tf.enable_eager_execution()
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """ rescales an image such that its pixels values are between 0
            and 1 and its largest side is 512 pixels
        Arg:
           - image: np.ndarray (h, w, 3) containing the image to be scaled
        Returns:
           - A scaled image Tensor
        """
        img_error = 'image must be a numpy.ndarray with shape (h, w, 3)'
        if not isinstance(image, np.ndarray):
            raise TypeError(img_error)
        if len(image.shape) != 3:
            raise TypeError(img_error)
        if image.shape[2] != 3:
            raise TypeError(img_error)

        h, w, _ = image.shape
        max_dim = 512
        maximum = max(h, w)
        scale = max_dim / maximum
        new_shape = (int(h*scale), int(w*scale))
        image = tf.expand_dims(image, axis=0)
        image = tf.image.resize_bicubic(image, new_shape)
        image = tf.cast(image, tf.float32)
        image /= 255
        image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=1)
        return image
