#!/usr/bin/env python3
"""
VAE(Variational Auto-Encoder)
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
    inputs = keras.layers.Input(shape=(input_dims,))

    # Encoder

    encoder = keras.layers.Dense(hidden_layers[0],
                                 activation='relu')(inputs)

    for i in range(1, len(hidden_layers)):
        encoder = keras.layers.Dense(hidden_layers[i],
                                     activation='relu')(encoder)

    z_mean = keras.layers.Dense(latent_dims, activation=None)(encoder)
    z_log_sigma = keras.layers.Dense(latent_dims, activation=None)(encoder)

    def sampling(args):
        """
        Parameters for sampling from a multivariate Gaussian
        """
        z_mean, z_log_var = args
        batch = keras.backend.shape(z_mean)[0]
        dim = keras.backend.int_shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon

    z = keras.layers.Lambda(sampling,
                            output_shape=(latent_dims, ))([z_mean,
                                                           z_log_sigma])

    latent_input = keras.layers.Input(shape=(latent_dims,))

    # Decoder

    decoder = keras.layers.Dense(hidden_layers[-1],
                                 activation='relu')(latent_input)

    for i in range(len(hidden_layers) - 2, -1, -1):
        decoder = keras.layers.Dense(hidden_layers[i],
                                     activation='relu')(decoder)

    output_decoder = keras.layers.Dense(input_dims,
                                        activation='sigmoid')(decoder)

    encoder = keras.Model(inputs, [z, z_mean, z_log_sigma])

    decoder = keras.Model(latent_input, output_decoder)

    outputs = decoder(encoder(inputs)[-1])

    vae = keras.Model(inputs, outputs)

    def vae_loss(x, x_decoder_mean):
        x_loss = keras.backend.binary_crossentropy(x, x_decoder_mean)
        x_loss = keras.backend.sum(x_loss, axis=1)
        kl_loss = - 0.5 * keras.backend.mean(1 + z_log_sigma -
                                             keras.backend.square(z_mean) -
                                             keras.backend.exp(z_log_sigma),
                                             axis=-1)
        return x_loss + kl_loss

    vae.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, vae
