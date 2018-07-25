import numpy as np

def get_n_columns(ndarray):
    return ndarray.shape[1]


def get_n_rows(ndarray):
    return ndarray.shape[0]


def drop_column(ndarray, column_idx):
    return np.delete(ndarray, column_idx, 1)


def drop_row(ndarray, row_idx):
    return np.delete(ndarray, row_idx, 0)
