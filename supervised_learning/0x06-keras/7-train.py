#!/usr/bin/env python3
"""
trains a model using mini-batch gradient descent
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, verbose=True, shuffle=False):
    """
    early_stopping is a boolean that indicates whether early
    stopping should be used
    early stopping should only be performed if
    validation_data exists
    early stopping should be based on validation loss
    patience is the patience used for early stoppin
    """
    callback = []
    if validation_data and early_stopping:
        early_stopping = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        )
        callback.append(early_stopping)

    if validation_data and learning_rate_decay:
        lr_decay = K.callbacks.LearningRateScheduler(
            schedule=lambda epoch: alpha /
            (1 + decay_rate * epoch)
        )
        callback.append(lr_decay)
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callback,
        verbose=verbose,
        shuffle=shuffle
    )
    return history
