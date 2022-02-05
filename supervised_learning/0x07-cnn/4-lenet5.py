#!/usr/bin/env python3
"""
Build a modified version of the
LeNet-5 architecture using tensorflow
"""
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """
    x is a tf.placeholder of shape (m, 28, 28, 1) containing
    the input images for the network
    m is the number of images
    y is a tf.placeholder of shape (m, 10) containing the one-hot
    labels for the network
    The model should consist of the following layers in order:
        Convolutional layer with 6 kernels of shape 5x5 with same padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Convolutional layer with 16 kernels of shape 5x5 with valid padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Fully connected layer with 120 nodes
        Fully connected layer with 84 nodes
        Fully connected softmax output layer with 10 nodes
    """
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)
    y_pred = tf.layer.conv2d(x, filters=6, kernel_size=(5, 5), padding='same',
                             activation='relu', kernel_initializer=initializer)

    y_pred = tf.layer.max_pooling2d(y_pred, pool_size=(2, 2), strides=(2, 2))

    y_pred = tf.layer.conv2d(y_pred, filters=16, kernel_size=(5, 5),
                             padding='valid', activation='relu',
                             kernel_initializer=initializer)

    y_pred = tf.layer.max_pooling2d(y_pred, pool_size=(2, 2), strides=(2, 2))

    y_pred = tf.layer.flatten(y_pred)

    y_pred = tf.layer.dense(y_pred, units=120, activation='relu',
                            kernel_initializer=initializer)

    y_pred = tf.layer.dense(y_pred, units=84, activation='relu',
                            kernel_initializer=initializer)

    y_pred = tf.layer.dense(y_pred, units=10, activation='softmax',
                            kernel_initializer=initializer)

    train = tf.train.AdamOptimizer()
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    train = train.minimize(loss)
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return y_pred, train, loss, accuracy
