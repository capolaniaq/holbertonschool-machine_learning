#!/usr/bin/env python3
"""
sets up Adam optimization for a keras model with
categorical crossentropy loss and accuracy metrics
"""
import tensorflow as tf


def optimize_model(network, alpha, beta1, beta2):
    """
    network is the model to optimize
    alpha is the learning rate
    beta1 is the first Adam optimization parameter
    beta2 is the second Adam optimization parameter
    """
    adam = tf.keras.optimizers.Adam(learning_rate=alpha,
                                    beta_1=beta1, beta_2=beta2)
    network.compile(
                    optimizer=adam,
                    loss=tf.keras.losses.CategoricalCrossentropy(),
                    metrics=[tf.keras.metrics.CategoricalAccuracy()])
