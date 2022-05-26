#!/usr/bin/env python3
"""
loads and preps a dataset for machine translation
"""

import tensorflow_datasets as tfds
import numpy as np


class Dataset:
    """
    Dataset class
    """

    def __init__(self):
        """
        Constructor class
        """
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validate', as_supervised=True)
        tok_pt, tok_en = self.tokenize_dataset(self.data_train)
        self.tokenizer_pt = tok_pt
        self.tokenizer_en = tok_en

    def tokenize_dataset(self, data):
        """
        Args:
        data: is a tf.data.Dataset whose examples are formatted
        as a tupla (pt, en)
            pt is the tf.Tensor containing the Portuguese sentence
            en is the tf.Tensor containing the corresponding English sentence

        The maximum vocab size should be set to 2**15

        Returns:
            tokenizer_pt is the Portuguese tokenizer
            tokenizer_en is the English tokenizer
        """
        tok_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
                       (pt.numpy() for pt, en in data), target_vocab_size=2**15)
        tok_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
                       (en.numpy() for pt, en in data), target_vocab_size=2**15)

        return tok_pt, tok_en

    def encode(self, pt, en):
        """
        Encodes a translation into tokens

        Args:
            pt is the tf.Tensor containing the Portuguese sentence
            en is the tf.Tensor containing the corresponding English sentence

        The tokenized sentences should include the start and end of
        sentence tokens
        The start token should be indexed as vocab_size
        The end token should be indexed as vocab_size + 1

        Returns:
            pt_tokens is a np.ndarray containing the Portuguese tokens
            en_tokens is a np.ndarray. containing the English tokens
        """
        pt_tokens = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
                                                                pt.numpy()) + [self.tokenizer_pt.vocab_size + 1]
        en_tokens = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
                                                                en.numpy()) + [self.tokenizer_en.vocab_size + 1]

        return np.array(pt_tokens), np.array(en_tokens)
