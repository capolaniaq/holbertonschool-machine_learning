#!/usr/bin/env python3
"""
RNN Decoder
"""

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    class RNN Decoder
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        Constructor class:
            vocab is an integer representing the size of the
                output vocabulary
            embedding is an integer representing the dimensionality
                of the embedding vector
            units is an integer representing the number of hidden
                units in the RNN cell
            batch is an integer representing the batch size
        """
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units, return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(vocab)

        super(RNNDecoder, self).__init__()

        def call(self, x, s_prev, hidden_states):
            """
            x is a tensor of shape (batch, 1) containing the previous word in the
                target sequence as an index of the target vocabulary
            s_prev is a tensor of shape (batch, units) containing the previous
                decoder hidden state
            hidden_states is a tensor of shape (batch, input_seq_len, units)
                containing the outputs of the encoder
            Returns:
                y is a tensor of shape (batch, vocab) containing the output word
                    as a one hot vector in the target vocabulary
                s is a tensor of shape (batch, units) containing the new decoder hidden state
            """
            batch, units = s_prev.shape
            attention = SelfAttention(units)
            context_vector, attention_weights = attention(s_prev, hidden_states)
            x = self.embedding(x)
            x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
            output, s = self.gru(x)
            output = tf.reshape(output, (output.shape[0], output.shape[2]))
            y = self.F(output)
            return y, s
