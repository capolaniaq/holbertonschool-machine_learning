#!/usr/bin/env python3
"""
VAE(Variational Auto-Encoder)
"""

import tensorflow.keras as keras


def sampling(args, size):
    """
    Parameters for sampling from a multivariate Gaussian
    """
    z_mean, z_log_sigma = args
    ep = keras.backend.random_normal(shape=(keras.backend.shape(z_mean)[0],
                                            size),
                                     mean=0., stddev=0.1)
    return z_mean + keras.backend.exp(z_log_sigma) * ep


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    input_dims is an integer containing the dimensions of the model input

    hidden_layers is a list containing the number of nodes for each hidden
    layer in the encoder, respectively
        the hidden layers should be reversed for the decoder

    latent_dims is an integer containing the dimensions of the latent space
    representation

    Returns: encoder, decoder, auto
        encoder is the encoder model, which should output the latent
        representation, the mean, and the log variance, respectively
        decoder is the decoder model
        auto is the full autoencoder model

    The autoencoder model should be compiled using adam optimization and
    binary cross-entropy loss
    All layers should use a relu activation except for the mean and log
    variance
    layers in the encoder, which should use None, and the last layer in
    the decoder,
    which should use sigmoid
    """
    input_encoder = keras.layers.Input(shape=(input_dims,))

    # Encoder

    encoder = keras.layers.Dense(hidden_layers[0],
                                 activation='relu')(input_encoder)

    for i in range(1, len(hidden_layers)):
        encoder = keras.layers.Dense(hidden_layers[i],
                                     activation='relu')(encoder)

    z_mean = keras.layers.Dense(latent_dims, activation=None)(encoder)
    z_log_sigma = keras.layers.Dense(latent_dims, activation=None)(encoder)

    z = keras.layers.Lambda(sampling)([z_mean, z_log_sigma], int(latent_dims))

    latent_input = keras.layers.Input(shape=(latent_dims,))

    # Decoder

    decoder = keras.layers.Dense(hidden_layers[-1],
                                 activation='relu')(latent_input)

    for i in range(len(hidden_layers) - 2, -1, -1):
        decoder = keras.layers.Dense(hidden_layers[i],
                                     activation='relu')(decoder)

    output_decoder = keras.layers.Dense(input_dims,
                                        activation='sigmoid')(decoder)

    encoder = keras.Model(input_encoder, [z_mean, z_log_sigma, z])

    decoder = keras.Model(latent_input, output_decoder)

    output = decoder(encoder(input_encoder)[2])
    auto = keras.Model(input_encoder, output)
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
