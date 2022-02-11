#!/usr/bin/env python3
"""
ResNet50 Architecture
"""

import tensorflow.keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    You can assume the input data will have shape (224, 224, 3)
    All convolutions inside and outside the blocks should be
    followed by batch normalization along the channels axis and
    a rectified linear activation (ReLU), respectively.
    All weights should use he normal initialization
    """
    input = K.Input(shape=(224, 224, 3))
    initializer = K.initializers.HeNormal()

    convol = K.layers.Conv2D(filters=64,
                             kernel_size=(7, 7),
                             strides=(2, 2),
                             padding='same',
                             kernel_initializer=initializer)(input)

    batch_norm = K.layers.BatchNormalization()(convol)

    activation = K.layers.Activation('relu')(batch_norm)

    max_pool = K.layers.MaxPooling2D(pool_size=(3, 3),
                                     strides=(2, 2),
                                     padding='same')(activation)

    project = projection_block(max_pool, [64, 64, 256], 1)

    identity = identity_block(project, [64, 64, 256])
    identity = identity_block(identity, [64, 64, 256])

    project = projection_block(identity, [128, 128, 512])

    identity = identity_block(project, [128, 128, 512])
    identity = identity_block(identity, [128, 128, 512])
    identity = identity_block(identity, [128, 128, 512])

    project = projection_block(identity, [256, 256, 1024])

    identity = identity_block(project, [256, 256, 1024])
    identity = identity_block(identity, [256, 256, 1024])
    identity = identity_block(identity, [256, 256, 1024])
    identity = identity_block(identity, [256, 256, 1024])
    identity = identity_block(identity, [256, 256, 1024])

    project = projection_block(identity, [512, 512, 2048])

    identity = identity_block(project, [512, 512, 2048])
    identity = identity_block(identity, [512, 512, 2048])

    avg = K.layers.AveragePooling2D(pool_size=(7, 7),
                                    padding='same')(identity)

    output = K.layers.Dense(units=1000,
                            activation='softmax',
                            kernel_initializer=initializer)(avg)

    model = K.models.Model(inputs=input, outputs=output)

    return model
