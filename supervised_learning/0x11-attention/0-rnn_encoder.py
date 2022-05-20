#!/usr/bin/env python3
"""
Create a class RNNEncoder
"""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    Class RNNEncoder
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Constructor class:
            vocab: is an integer representing the size of the output vocabulary
            embedding: is an integer representing the dimensionality of the
                embedding vector
            units: is an integr representing the number of hidden units in the
                RNN cell
            batch: is an integer representing the batch size
        """
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units, return_sequences=True,
                                return_state=True, recurrent_initializer='glorot_uniform')
        super(RNNEncoder, self).__init__()

    def initialize_hidden_state(self):
        """
        Initialize the hidden state for the RNN
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        x: is a tensor of shape (batch, input_seq_len, d)
        initial: is the initial hidden state
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
