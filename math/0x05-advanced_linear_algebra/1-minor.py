#!/usr/bin/env python3
"""
Module that calculate the minor of a matrix
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


def minor(matrix):
    """
    Return a minor from matrix
    """
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    if [x for x in matrix if type(x) is not list]:
        raise TypeError("matrix must be a list of lists")

    if [x for x in matrix if len(x) != len(matrix)]:
        raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return [[1]]

    n = len(matrix)
    minor = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == 0 and j == 0:
                row.append(matrix[1][1])
            elif i == 0 and j == 1:
                row.append(matrix[1][0])
            elif i == 1 and j == 0:
                row.append(matrix[0][1])
            else:
                row.append(matrix[0][0])
        minor.append(row)

    if len(matrix) > 2:
        for i in range(n):
            for j in range(n):
                det = []
                for k, row in enumerate(matrix):
                    if k != i:
                        new_row = row.copy()
                        new_row.pop(j)
                        det.append(new_row)
                minor[i][j] = determinant(det)

    return minor
