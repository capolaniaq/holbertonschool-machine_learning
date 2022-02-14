#!/usr/bin/env python3
"""
Desnet121 model for Keras.
"""

import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    growth_rate is the growth rate
    compression is the compression factor
    You can assume the input data will have shape (224, 224, 3)
    All convolutions should be preceded by Batch Normalization
    and a rectified linear activation (ReLU), respectively
    All weights should use he normal initialization
    """
    init = K.initializers.HeNormal()
    input = K.Input(shape=(224, 224, 3))

    batch_norm = K.layers.BatchNormalization()(input)
    activation = K.layers.Activation('relu')(batch_norm)

    conv_1 = K.layers.Conv2D(filters=2*growth_rate,
                             kernel_size=(7, 7),
                             strides=(2, 2),
                             padding='same',
                             kernel_initializer=init)(activation)

    max_pool = K.layers.MaxPool2D(pool_size=(3, 3),
                                  strides=(2, 2),
                                  padding='same')(conv_1)

    dense_block_1, nb_filters = dense_block(max_pool, 2*growth_rate,
                                            growth_rate, 6)

    transition_layer_1, nb_filters = transition_layer(dense_block_1,
                                                      nb_filters,
                                                      compression)

    dense_block_2, nb_filters = dense_block(transition_layer_1,
                                            nb_filters, growth_rate, 12)

    transition_layer_2, nb_filters = transition_layer(dense_block_2,
                                                      nb_filters,
                                                      compression)

    dense_block_3, nb_filters = dense_block(transition_layer_2,
                                            nb_filters, growth_rate, 24)

    transition_layer_3, nb_filters = transition_layer(dense_block_3,
                                                      nb_filters,
                                                      compression)

    dense_block_4, nb_filters = dense_block(transition_layer_3,
                                            nb_filters, growth_rate, 16)

    avg = K.layers.AveragePooling2D(pool_size=(7, 7),
                                    strides=(1, 1),
                                    padding='valid')(dense_block_4)

    output = K.layers.Dense(units=1000, activation='softmax',
                            kernel_initializer=init)(avg)

    model = K.models.Model(inputs=input, outputs=output)

    return model
