#!/usr/bin/env python3
"""
Module that calculate a determinant form matrix
"""

def determinant_3x3(matrix):
    """
    Function that to get a matrix 3 x 3
    and return the determinant for this
    """
    determinant = 0
    if len(matrix) == 3:
        for i, value in enumerate(matrix[0]):
            if i == 0:
                a = matrix[1][1] * matrix[2][2]
                b = matrix[1][2] * matrix[2][1]
                determinant = determinant + (value * (a - b))
            elif i == 1:
                a = matrix[1][0] * matrix[2][2]
                b = matrix[1][2] * matrix[2][0]
                determinant = determinant - (value * (a - b))
            elif i == 2:
                a = matrix[1][0] * matrix[2][1]
                b = matrix[1][1] * matrix[2][0]
                determinant = determinant + (value * (a - b))
    return determinant


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
    if len(matrix) == 3:
        return determinant_3x3(matrix)

    matrix_3x3 = []
    matrix_copy = matrix.copy()
    row = matrix[0].copy()
    if len(matrix) == 4:
        for i, value in enumerate(row):
            row1 = matrix[1].copy()
            row2 = matrix[2].copy()
            row3 = matrix[3].copy()
            row1.pop(i)
            row2.pop(i)
            row3.pop(i)
            matrix_3x3.append(row1)
            matrix_3x3.append(row2)
            matrix_3x3.append(row3)
            if i == 0:
                determinant = determinant + (value * determinant_3x3(matrix_3x3))
            elif i == 1:
                determinant = determinant - (value * determinant_3x3(matrix_3x3))
            elif i == 2:
                determinant = determinant + (value * determinant_3x3(matrix_3x3))
            else:
                determinant = determinant - (value * determinant_3x3(matrix_3x3))
            matrix_copy = matrix.copy()
            matrix_3x3.clear()



    return determinant
