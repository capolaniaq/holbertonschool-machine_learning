#!/usr/bin/env python3
"""
Creates autoencoder
"""

import numpy as np
import tensorflow as tf


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    input_dims is an integer containing the dimensions of the model input
    hidden_layers is a list containing the number of nodes for each hidden
    layer in the encoder, respectively
        the hidden layers should be reversed for the decoder
    latent_dims is an integer containing the dimensions of the latent space
    representation
    Returns: encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the full autoencoder model
    The autoencoder model should be compiled using adam optimization and
    binary cross-entropy loss
    All layers should use a relu activation except for the last layer in the
    decoder, which should use sigmoid
    """
    input = tf.keras.Input(shape=(input_dims, ))
    encoder_layer = input

    for i in range(len(hidden_layers)):
        encoder_layer = tf.keras.layers.Dense(hidden_layers[i],
                                              activation='relu')(encoder_layer)

    laten_layer = tf.keras.layers.Dense(latent_dims,
                                        activation='relu')(encoder_layer)

    decoder_layer = laten_layer

    for i in range(len(hidden_layers) - 1, -1, -1):
        decoder_layer = tf.keras.layers.Dense(hidden_layers[i],
                                              activation='relu')(decoder_layer)

    decoder_layer = tf.keras.layers.Dense(input_dims,
                                          activation='sigmoid')(decoder_layer)

    encoder = tf.keras.Model(input, laten_layer)
    decoder = tf.keras.Model(laten_layer, decoder_layer)
    auto = tf.keras.Model(input, decoder_layer)
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
