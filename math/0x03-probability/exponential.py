#!/usr/bin/env python3
"""
    Exponencial distribution
"""


class Exponential:
    """
        Class Exponencial
    """

    def __init__(self, data=None, lambtha=1.):
        """
            Constructor
        """
        if data is None:
            if lambtha < 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha) + 1
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = float(1/(sum(data) / len(data)))
