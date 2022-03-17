#!/usr/bin/env python3
"""
Module that calculate a determinant form matrix
"""


def determinant(matrix):
    """
    Calculate the determinant of a matrix
    """
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

    n = len(matrix)

    determinant = 1
    for i in range(n - 1):
        for j in range(i + 1, n):
            if matrix[i][i] == 0:
                determinant = determinant * - 1
                row_tmp = matrix[i].copy()
                matrix[i] = matrix[j]
                matrix[j] = row_tmp
            factor = matrix[j][i] / matrix[i][i]
            for k in range(n):
                matrix[j][k] = matrix[j][k] - (factor * matrix[i][k])


    for i in range(n):
        determinant *= matrix[i][i]

    return round(determinant)
