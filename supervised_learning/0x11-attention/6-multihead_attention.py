#!/usr/bin/env python3
"""
Multi-Head Attention
"""

import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    MultiHeadAttention class
    """
    def __init__(self, dm, h):
        """
        dm is an integer representing the dimensionality of the model
        h is an integer representing the number of heads
        dm is divisible by h
        Sets the following public instance attributes:
            h - the number of heads
            dm - the dimensionality of the model
            depth - the depth of each attention head
            Wq - a Dense layer with dm units, used to generate the query matrix
            Wk - a Dense layer with dm units, used to generate the key matrix
            Wv - a Dense layer with dm units, used to generate the value matrix
            linear - a Dense layer with dm units, used to generate the attentio
            output
        """
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

        super(MultiHeadAttention, self).__init__()

    def call(self, Q, K, V, mask):
        """
        Q is a tensor of shape (batch, seq_len_q, dk)
            containing the input to generate the query matrix
        K is a tensor of shape (batch, seq_len_v, dk)
            containing the input to generate the key matrix
        V is a tensor of shape (batch, seq_len_v, dv)
            containing the input to generate the value matrix
        mask is always None
        Returns: output, weights
        outputa tensor with its last two dimensions as
            (..., seq_len_q, dm) containing the scaled dot product attention
        weights a tensor with its last three dimensions as
            (..., h, seq_len_q, seq_len_v) containing the attention weights
        """
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        Q = tf.concat(tf.split(Q, self.h, axis=-1), axis=0)
        K = tf.concat(tf.split(K, self.h, axis=-1), axis=0)
        V = tf.concat(tf.split(V, self.h, axis=-1), axis=0)

        output, weights = sdp_attention(Q, K, V, mask)

        output = tf.concat(tf.split(output, self.h, axis=0), axis=-1)
        output = self.linear(output)

        return output, weights
