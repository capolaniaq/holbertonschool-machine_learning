#/usr/bin/env python3
"""
Make a Dataframe from numpy array
"""

import pandas as pd


def from_numpy(array):
    """
    Creates a pd.Dataframe from a ndarray

    Args:
        array is the numpy array
    Return:
        the newly created pd.DataFrame
    """
    alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    col = array.shape[1]
    df = pd.DataFrame(array, columns=alphabet[0:col])
    return df
