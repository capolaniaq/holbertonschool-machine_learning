#!/usr/bin/env python3
"""load dataframe from file"""

import pandas as pd


def from_file(filename, delimiter):
    """
    Load data from a file and convert to pd.DataFrame

    Args:
        filename is a name of the file to load
        delimiter is the column separator
    Return:
        the loaded DataFrame
    """
    df = pd.read_csv(filename, delimiter=delimiter)
    return df
