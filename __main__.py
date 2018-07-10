import sys
from collections import namedtuple

import pandas as pd

from Perceptron import Perceptron


Data = namedtuple("Data", ['features', 'assigned', 'feature_names'])


def drop_column(data_frame, col_idx):
    return data_frame.drop(col_idx, axis=1)


def drop_row(data_frame, row_idx):
    return data_frame.drop(row_idx, axis=0)


def get_column(data_frame, col_idx):
    return data_frame.iloc[:, col_idx]


def get_row(data_frame, row_idx):
    return data_frame.iloc[row_idx, :]


def load_data(features_path):
    csv_data = pd.read_csv(features_path, header=None, low_memory=False)

    assigned = get_column(csv_data, len(csv_data.columns) - 1)
    assigned = assigned.iloc[1:]
    csv_data = drop_column(csv_data, len(csv_data.columns) - 1)

    feature_names = get_row(csv_data, 0)
    csv_data = drop_row(csv_data, 0)

    return Data(csv_data.values, assigned, feature_names)


def get_n_columns(ndarray):
    return ndarray.shape[1]


if __name__ == '__main__':
    features_path = sys.argv[1]

    data = load_data(features_path)

    perceptron = Perceptron(get_n_columns(data.features), data.feature_names)
