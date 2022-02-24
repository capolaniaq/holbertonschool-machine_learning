#!/usr/bin/env python3
"""
Transfer Learning with ResNet50
"""

import numpy as np
import tensorflow.keras as K
import tensorflow as tf


def preprocess_data(X, Y):
    """
    Preprocess data for use in model keras
    """
    X_p = K.applications.resnet50.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, num_classes=10)
    return X_p, Y_p


if __name__ == '__main__':

    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()

    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    base_model = K.applications.resnet50.ResNet50(include_top=False,
                                                  input_shape=(224, 224, 3))

    inputs = K.Input(shape=(32, 32, 3))
    input = K.layers.Lambda(lambda image : tf.image.resize(image, (224, 224)))(inputs)


    x = base_model(input, training=False)

    x = K.layers.GlobalAveragePooling2D()(x)

    x = K.layers.Dense(1000, activation='relu')(x)

    x = K.layers.Dropout(0.5)(x)

    outputs = K.layers.Dense(10, activation='softmax')(x)

    model = K.models.Model(inputs=inputs, outputs=outputs)

    base_model.trainable = False

    optimizer = K.optimizers.Adam()

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.fit(X_train, Y_train,
              validation_data=(X_test, Y_test),
              batch_size=128,
              epochs=4,
              verbose=1)

    for layer in base_model.layers[:87]:
        layer.trainable = False

    for layer in base_model.layers[87:]:
        layer.trainable = True

    optimizer = K.optimizers.Adam(1e-5)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.fit(X_train, Y_train,
              validation_data=(X_test, Y_test),
              batch_size=300, epochs=4, verbose=1)
