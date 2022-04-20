#!/usr/bin/env python3
"""
Creates autoencoder
"""

import tensorflow.keras as keras


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
    input_layer = keras.Input(shape=(input_dims,))
    hidden_layers = [input_layer] + [keras.layers.Dense(units=layer,
                                                        activation='relu')
                                        for layer in hidden_layers]
    encoder = keras.models.Sequential(hidden_layers)
    decoder_layers = [keras.layers.Dense(units=layer, activation='relu')
                        for layer in reversed(hidden_layers[1:])]
    decoder_layers.append(keras.layers.Dense(units=input_dims,
                                                activation='sigmoid'))
    decoder = keras.models.Sequential(decoder_layers)
    auto = keras.models.Sequential([encoder, decoder])
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
