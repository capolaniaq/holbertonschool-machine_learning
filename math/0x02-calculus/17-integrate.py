#!/usr/bin/env python3
"""
Integrate.
"""


def poly_integral(poly, C=0):
    """
        Function that calculates the integral from a function
    """
    if type(poly) is not list or type(C) is not (int, float):
        return None
    integrate = [C]
    for x, variable in enumerate(poly):
        if x == 0:
            integrate.append(variable)
        elif variable == 0:
            integrate.append(0)
        else:
            integrate.append(variable / (x + 1))
    return integrate
