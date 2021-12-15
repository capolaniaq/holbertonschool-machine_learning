#!/usr/bin/env python3
"""
Poisson distribution
"""


class Poisson:
    """
        class Poisson:
    """

    pi = 3.1415926536
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """
        Class constructor
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculates the value of the PMF
        """
        from math import factorial
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        return (self.e**-self.lambtha)*(self.lambtha**k)/factorial(k)
