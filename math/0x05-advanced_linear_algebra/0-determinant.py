#!/usr/bin/env python3
"""
Module that calculate a determinant form matrix
"""


def determinant(matrix):
    """Function that calculate a determinant from matrix"""
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1

    for row in matrix:
        if len(matrix) != len(row):
            raise ValueError("matrix must be a square matrix")

    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    determinant = 0
    if len(matrix) == 3:
        for i, value in enumerate(matrix[0]):
            if i == 0:
                determinant = (value * ((matrix[1][1] * matrix[2][2]) - (matrix[1][2] * matrix[2][1])))
            elif i == 1:
                determinant = determinant - (value * ((matrix[1][0] * matrix[2][2]) - (matrix[1][2] * matrix[2][0])))
            else:
                determinant = determinant + (value * ((matrix[1][0] * matrix[2][1]) - (matrix[1][1] * matrix[2][0])))
    return determinant
