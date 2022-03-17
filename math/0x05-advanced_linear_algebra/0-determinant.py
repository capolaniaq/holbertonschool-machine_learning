#!/usr/bin/env python3
"""
Module that calculate a determinant form matrix
"""


def minor(matrix, row, col):
    """Function that calculate a minor from matrix"""
    minor = []
    for i in range(len(matrix)):
        if i != row:
            minor.append([])
            for j in range(len(matrix[i])):
                if j != col:
                    minor[-1].append(matrix[i][j])
    return minor

def determinant(matrix):
    """Function that calculate a determinant from matrix"""
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    if [x for x in matrix if type(x) is not list]:
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1

    if [row for row in matrix if len(row) != len(matrix)]:
        raise ValueError("matrix must be a square matrix")

    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    determinant = 0
    if len(matrix) > 2:
        for i in range(len(matrix)):
            determinant += matrix[0][i] * determinant(minor(matrix, 0, i)) * (-1) ** i
    return determinant
