import sys

import pandas as pd


def drop_column(data_frame, col_idx):
    return data_frame.drop(col_idx, axis=1)


def get_column(data_frame, col_idx):
    return data_frame.iloc[:, col_idx]


def load_data(features_path):
    csv_data = pd.read_csv(features_path, header=None, low_memory=False)

    assigned = get_column(csv_data, len(csv_data.columns) - 1)

    csv_data = drop_column(csv_data, len(csv_data.columns) - 1)

    return csv_data.as_matrix(), assigned


if __name__ == '__main__':
    features_path = sys.argv[1]

    data = load_data(features_path)
