#!/usr/bin/env python3
"""
This script is used to calculate the unigram BLEU score.
"""

import numpy as np


def uni_bleu(references, sentence):
    """
    Calculate the unigram BLEU score.
    :param references: a list of reference sentences
    :param sentence: a hypothesis sentence
    :return: the unigram BLEU score
    """
    # Calculate the length of the hypothesis sentence.
    hyp_len = len(sentence)

    # Calculate the unigram BLEU score.
    score = 0.
    for ref in references:
        ref_len = len(ref)
        # Calculate the length of the common sub-sequence.
        lcs = len(np.intersect1d(sentence, ref))
        score += lcs / (hyp_len + ref_len - lcs)
    score /= len(references)

    return score
