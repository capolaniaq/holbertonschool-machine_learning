#!/usr/bin/env python3
"""
Self-Attention
"""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    class selfAttention
    """
    def __init__(self, units):
        """
        Constructor class:
            units: is an integer representing the number
            of hidden units in the
        """
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

        super(SelfAttention, self).__init__()

    def call(self, s_prev, hidden_states):
        """
        s_prev is a tensor of shape (batch, units)
        hidden_states is a tensor of shape (batch, input_seq_len, units)
        """
        s_prev = tf.expand_dims(s_prev, 1)
        score = tf.nn.tanh(self.W(s_prev) + self.U(hidden_states))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * hidden_states
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights
