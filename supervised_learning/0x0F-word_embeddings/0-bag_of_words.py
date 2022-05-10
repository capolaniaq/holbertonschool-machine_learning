#!/usr/bin/env python3
"""
Function that make a bag of words from a list of sentences
"""

from sklearn.feature_extraction.text import CountVectorizer


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
    if vocab is None:
        vocab = []
        vectorizer = CountVectorizer()
    else:
        vocab = vocab
        vectorizer = CountVectorizer(vocabulary=vocab)

    embeddings = vectorizer.fit_transform(sentences).toarray()
    features = list(vectorizer.get_feature_names_out())

    return embeddings, features
