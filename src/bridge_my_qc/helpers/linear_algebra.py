import numpy as np


def get_column_vectors(matrix):
    return [matrix[:, i] for i in range(matrix.shape[1])]


def get_row_vectors(matrix):
    return [matrix[i] for i in range(matrix.shape[0])]


def isSingular(matrix):
    det = np.linalg.det(matrix)
    print(f"Determinant is {det}")
    return not bool(det)
