#!/usr/bin/env python3
"""
trains a model using mini-batch gradient descent
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """
    early_stopping is a boolean that indicates whether early
    stopping should be used
    early stopping should only be performed if
    validation_data exists
    early stopping should be based on validation loss
    patience is the patience used for early stoppin
    """
    callback = None
    if validation_data != None and early_stopping != False:
        callback = [K.callbacks.EarlyStopping(
            monitor='loss',
            patience=patience
        )]
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=[callback],
        verbose=verbose,
        shuffle=shuffle
    )
    return history
