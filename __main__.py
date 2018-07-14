import sys

import InputParser
from KFolds import kfolds
from Perceptron import Perceptron


def get_n_columns(ndarray):
    return ndarray.shape[1]


def get_n_rows(ndarray):
    return ndarray.shape[0]


def cross_validate(data, n_folds, n_iterations):
    # TODO Should I make it so you can pass in a ML algorithm?
    # TODO Should this be a part of the perceptron?
    sum_accuracy = 0

    perceptron = Perceptron(get_n_columns(data.features), data.feature_names)

    folds = kfolds(get_n_rows(data.features), n_folds)

    for fold in folds:
        perceptron.train(fold.train_index, data, n_iterations)
        accuracy = perceptron.test(fold.test_index, data)
        sum_accuracy += accuracy

        print('Accuracy: %.9f' % accuracy)

    print('Average accuracy: %.9f' % (sum_accuracy / n_folds))


if __name__ == '__main__':
    features_path = sys.argv[1]

    n_folds = 10
    n_iterations = 10

    data = InputParser.load_csv_data(features_path)

    cross_validate(data, n_folds, n_iterations)

