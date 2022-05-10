#!/usr/bin/env python3
"""
Function that make a bag of words from a list of sentences
"""

import numpy as np


def bag_of_words(sentences, vocab=None):
    """
    sentences is a list of sentences to analyze
    vocab is a list of the vocabulary words to use for the analysis
    If None, all words within sentences should be used
    Returns: embeddings, features
        embeddings is a numpy.ndarray of shape (s, f) containing the embeddings
        s is the number of sentences in sentences
        f is the number of features analyzed
        features is a list of the features used for embeddings
    """
    for i, sentence in enumerate(sentences):
        sentence = sentence.split()
        for j, word in enumerate(sentence):
            word = word.lower()
            new_word = ''
            for k, letter in enumerate(word):
                if letter.isalpha() is False:
                    break
                new_word = new_word + str(letter)
            sentence[j] = new_word
        sentence = ' '.join([str(item) for item in sentence])
        sentences[i] = sentence

    if vocab is None:
        vocab = set()
        for sentence in sentences:
            vocab.update(sentence.split())
        vocab = list(vocab)
    vocab.sort()

    s = len(sentences)
    f = len(vocab)

    embeddings = np.zeros((s, f))

    for i in range(s):
        for j in range(f):
            embeddings[i, j] = sentences[i].count(vocab[j])

    return embeddings, vocab
