#!/usr/bin/env python3
"""
This script is used to calculate the unigram BLEU score.
"""

import numpy as np


def uni_bleu(references, sentence):
    """
    Calculate the unigram BLEU score.
    param references: a list of reference sentences
    param sentence: a hypothesis sentence
    return: the unigram BLEU score
    """
    list_references = list(set(sentence))

    count_references = {}

    for reference in references:
        for word in reference:
            if word in list_references:
                if word not in count_references:
                    count_references[word] = reference.count(word)
                else:
                    if reference.count(word) > count_references[word]:
                        count_references[word] = reference.count(word)
                    else:
                        pass

    lenght_refrence = []
    for reference in references:
        lenght_refrence.append(len(reference))

    r = min(lenght_refrence)

    c = len(list_references)

    if c > r:
        bp = 1
    else:
        bp = np.exp(1 - r / c)

    blue = bp * np.exp(np.log(sum(count_references.values()) / c))

    return blue
