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
    elif poly == []:
        return None
    elif poly == [0]:
        return [C]

    integrate = [C]

    for x, variable in enumerate(poly):
        if type(variable) is not int and type(variable) is not float:
            return None

        if variable == 0:
            integrate.append(0)
        else:
            coeficient = variable / (x + 1)
            if coeficient % 1 == 0:
                integrate.append(int(coeficient))
            else:
                integrate.append(coeficient)

    return integrate
