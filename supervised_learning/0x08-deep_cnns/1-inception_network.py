#!/usr/bin/env python3
"""
Inception Network
"""

import tensorflow.keras as K

inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    You can assume the input data will have shape (224, 224, 3)
    All convolutions inside and outside the inception block should
    use a rectified linear activation (ReLU)
    Returns: the keras model
    """
    input = K.Input(shape=(224, 224, 3))

    conv1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                            padding='same', activation='relu')(input)

    pool1 = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(conv1)

    conv2_R = K.layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same',
                            activation='relu')(pool1)

    conv2 = K.layers.Conv2D(filters=192, kernel_size=(3, 3), padding='same',
                            activation='relu')(conv2_R)

    pool2 = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(conv2)

    inception1 = inception_block(pool2, filters=(64, 96, 128, 16, 32, 32))

    inception2 = inception_block(inception1, filters=(128, 128, 192, 32, 96, 64))

    pool3 = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(inception2)

    inception3 = inception_block(pool3, filters=(192, 96, 208, 16, 48, 64))

    inception4 = inception_block(inception3, filters=(160, 112, 224, 24, 64, 64))

    inception5 = inception_block(inception4, filters=(128, 128, 256, 24, 64, 64))

    inception6 = inception_block(inception5, filters=(112, 144, 288, 32, 64, 64))

    inception7 = inception_block(inception6, filters=(256, 160, 320, 32, 128, 128))

    pool4 = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(inception7)

    inception8 = inception_block(pool4, filters=(256, 160, 320, 32, 128, 128))

    inception9 = inception_block(inception8, filters=(384, 192, 384, 48, 128, 128))

    avg = K.layers.AveragePooling2D((7, 7), strides=(7, 7), padding='same')(inception9)

    drop = K.layers.Dropout(0.4)(avg)

    out = K.layers.Flatten()(drop)

    softmax = K.layers.Dense(units=1000, activation='softmax')(out)

    model = K.models.Model(inputs=input, outputs=softmax)

    return model








