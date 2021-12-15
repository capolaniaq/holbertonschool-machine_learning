#!/usr/bin/env python3
"""
Normal distribution
"""


class Normal:
    """
    Class Normal
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Constructor
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                mean = sum(data) / len(data)
                sum_stddev = 0
                for i in data:
                    sum_stddev += (i - mean) ** 2
                stddev = (sum_stddev / len(data)) ** (1/2)
                self.mean = float(mean)
                self.stddev = float(stddev)


    def z_score(self, x):
        """
        z-score of a given x-value
        """
        return float((x - self.mean)/self.stddev)

    def x_value(self, z):
        """
        x_value of a given z_score
        """
        return float(self.stddev*z + self.mean)
