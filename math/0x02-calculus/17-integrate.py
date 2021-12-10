#!/usr/bin/env python3
"""
Integrate.
"""


def poly_integral(poly, C=0):
    """
        Function that calculates the integral from a function
    """
    if poly is None or type(poly) is not list or type(C) is not int:
        return None
    integrate = [C]
    for x, variable in enumerate(poly):
        if x == 0:
            integrate.append(variable)
        else:
            integrate.append(variable / (x + 1))
    return integrate
