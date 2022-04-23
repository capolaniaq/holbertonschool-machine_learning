#!/usr/bin/env python3
"""
Convolutional Autoencoder
"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    input_dims is a tuple of integers containing the
    dimensions of the model input

    filters is a list containing the number of filters
    for each convolutional layer in the encoder, respectively
        the filters should be reversed for the decoder

    latent_dims is a tuple of integers containing the dimensions
    of the latent space representation

    Each convolution in the encoder should use a kernel size of (3, 3)
    with same padding and relu activation, followed by max pooling of
    size (2, 2)
    Each convolution in the decoder, except for the last two, should
    use a filter size of (3, 3) with same padding and relu activation,
    followed by upsampling of size (2, 2)
    The second to last convolution should instead use valid padding
    The last convolution should have the same number of filters as the
    number of channels in input_dims with sigmoid activation and no upsampling
    Returns: encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the full autoencoder model
    The autoencoder model should be compiled using adam optimization
    and binary cross-entropy loss
    """
    input_encoder = keras.layers.Input(shape=input_dims)

    # Encoder

    convol = keras.layers.Conv2D(filters=filters[0], kernel_size=(3, 3),
                                 padding='same',
                                 activation='relu')(input_encoder)
    pool = keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(convol)

    for i in range(1, len(filters)):
        convol = keras.layers.Conv2D(filters=filters[i], kernel_size=(3, 3),
                                     padding='same', activation='relu')(pool)
        pool = keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(convol)

    bottleneck = pool

    encoder = keras.Model(inputs=input_encoder, outputs=bottleneck)

    # Decoder

    input_decoder = keras.layers.Input(shape=latent_dims)

    convol = keras.layers.Conv2D(filters=filters[-1], kernel_size=(3, 3),
                                 padding='same',
                                 activation='relu')(input_decoder)

    pool = keras.layers.UpSampling2D(size=(2, 2))(convol)

    for i in range(len(filters) - 2, 0, -1):
        convol = keras.layers.Conv2D(filters=filters[i], kernel_size=(3, 3),
                                     padding='same',
                                     activation='relu')(pool)
        pool = keras.layers.UpSampling2D(size=(2, 2))(convol)

    convol = keras.layers.Conv2D(filters=filters[0], kernel_size=(3, 3),
                                 padding='valid',
                                 activation='relu')(pool)

    pool = keras.layers.UpSampling2D(size=(2, 2))(convol)

    output = keras.layers.Conv2D(filters=input_dims[-1], kernel_size=(3, 3),
                                 padding='same',
                                 activation='sigmoid')(pool)

    decoder = keras.Model(inputs=input_decoder, outputs=output)

    # Autoencoder

    auto = keras.Model(inputs=input_encoder,
                       outputs=decoder(encoder(input_encoder)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
