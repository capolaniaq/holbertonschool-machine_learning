#!/usr/bin/env python3
"""
Creates autoencoder
"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    input_dims is an integer containing the dimensions of the model input

    hidden_layers is a list containing the number of nodes for each hidden
    layer in the encoder, respectively
        the hidden layers should be reversed for the decoder

    latent_dims is an integer containing the dimensions of the latent
    space representation

    lambtha is the regularization parameter used for L1 regularization
    on the encoded output

    Returns: encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the sparse autoencoder model

    The sparse autoencoder model should be compiled using adam optimization
    and binary cross-entropy loss

    All layers should use a relu activation except for the last layer in
    the decoder, which should use sigmoid
    """
    input_encoder = keras.Input(shape=(input_dims,))
    l1 = keras.regularizers.L1(lambtha)

    # Encoder
    encoder = input_encoder
    for i in range(len(hidden_layers)):
        encoder = keras.layers.Dense(hidden_layers[i],
                                     activation='relu',
                                     activity_regularizer=l1)(encoder)

    latent = keras.layers.Dense(latent_dims,
                                activation='relu',
                                activity_regularizer=l1)(encoder)
    encoder = keras.Model(inputs=input_encoder, outputs=latent)

    # Decoder

    input_decoder = keras.Input(shape=(latent_dims,))
    decoder = input_decoder
    for i in range(len(hidden_layers) - 1, -1, -1):
        decoder = keras.layers.Dense(hidden_layers[i],
                                     activation='relu',
                                     activity_regularizer=l1)(decoder)

    output_decoder = keras.layers.Dense(input_dims,
                                        activation='sigmoid',
                                        activity_regularizer=l1)(decoder)

    decoder = keras.Model(inputs=input_decoder, outputs=output_decoder)

    # Autoencoder
    auto = keras.Model(inputs=input_encoder,
                       outputs=decoder(encoder(input_encoder)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
