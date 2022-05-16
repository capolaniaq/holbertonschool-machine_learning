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
    uni_sentence = list(set(sentence))

    count_references = {}

    for reference in references:
        for word in reference:
            if word in uni_sentence:
                if word not in count_references:
                    count_references[word] = reference.count(word)
                else:
                    if reference.count(word) > count_references[word]:
                        count_references[word] = reference.count(word)
                    else:
                        pass

    c = len(uni_sentence)

    list_references = []

    for reference in references:
        ren = len(reference)
        list_references.append((abs(ren - c)), ren)

    r = sorted(list_references, key=lambda x: x[0])[0][1]
    r = r[0][1]

    if c > r:
        bp = 1
    else:
        bp = np.exp(1 - r / c)

    blue = bp * np.exp(np.log(sum(count_references.values()) / c))

    return blue
