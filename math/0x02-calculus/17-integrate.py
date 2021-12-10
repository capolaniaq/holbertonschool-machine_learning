#!/usr/bin/env python3
"""
Integrate.
"""


def poly_integral(poly, C=0):
    """
        Function that calculates the integral from a function
    """
    if type(poly) is not list or type(C) not in (int, float):
        return None
    elif poly == [0]:
        return [C]
    elif poly == []:
        return None
    integrate = [C]
    for x, variable in enumerate(poly):
        if variable == 0:
            integrate.append(0)
        elif x != 0:
            coeficient = variable / (x + 1)
            integrate.append(coeficient)
        else:
            integrate.append(variable)
    return integrate
