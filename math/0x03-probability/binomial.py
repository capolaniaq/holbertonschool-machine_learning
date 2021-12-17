#!/usr/bin/env python3
"""
    Bimodal Distribution
"""


class Binomial:
    """
        Bimodal class
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Constructor
        """
        if type(data) is not list:
            if n <= 0:
                raise ValueError("n must be a positive value")
            elif p <= 0 or p > 1:
                raise ValueError("p must be greater than 0 and less than 1")
            else:
                self.n = n
                self.p = p
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                mean = sum(data) / len(data)
                variance = 0
                for i in data:
                    variance += (i - mean) ** 2
                variance = variance/len(data)
                self.p = 1 - (variance / mean)
                self.n = round(mean / self.p)
                self.p = mean / self.n
