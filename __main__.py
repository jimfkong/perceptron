import sys

import InputParser
from Perceptron import Perceptron


def get_n_columns(ndarray):
    return ndarray.shape[1]


if __name__ == '__main__':
    features_path = sys.argv[1]

    data = InputParser.load_csv_data(features_path)

    perceptron = Perceptron(get_n_columns(data.features), data.feature_names)
