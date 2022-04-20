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
    input_encoder = keras.Input(shape=(input_dims,))

    # Encoder
    encoder_layer = input_encoder
    for i in range(len(hidden_layers)):
        encoder_layer = keras.layers.Dense(hidden_layers[i],
                                           activation='relu')(encoder_layer)

    latent = keras.layers.Dense(latent_dims,
                                activation='relu')(encoder_layer)
    encoder = keras.Model(inputs=input_encoder, outputs=latent)

    # Decoder

    input_decoder = keras.Input(shape=(latent_dims,))
    decoder_layer = input_decoder
    for i in range(len(hidden_layers) - 1, -1, -1):
        decoder_layer = keras.layers.Dense(hidden_layers[i],
                                           activation='relu')(decoder_layer)

    output_decoder = keras.layers.Dense(input_dims,
                                        activation='sigmoid')(decoder_layer)

    decoder = keras.Model(inputs=input_decoder, outputs=output_decoder)

    # Autoencoder
    auto = keras.Model(inputs=input_encoder,
                       outputs=decoder(encoder(input_encoder)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
