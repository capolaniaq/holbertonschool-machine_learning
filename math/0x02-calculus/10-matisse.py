#!/usr/bin/env python3
"""
    Module that calculates the derivade from a function
"""


def poly_derivative(poly):
    """
        Function that calculates the derivade from a function
    """
    if type(poly) is not list:
        return None
    if len(poly) == 0:
        return None
    derivade = []
    for x, variable in enumerate(poly):
        if x > 0:
            derivade.append(x * variable)
    return derivade