from __future__ import division

import os

from enum import IntEnum
from sklearn.cross_validation import KFold
from FeatureHelpers import get_csv_writer
import pandas as pd
import numpy as np
import copy
import sys

#TODO Make a nicer way of handling classes
class Classification(IntEnum):
    """
    Enum containing possible categories.
    """
    '''
    Commerce = 0
    Construction = 1
    Education = 2
    Environment = 3
    Government = 4
    Health = 5
    Hospitality = 6
    Humanities = 7
    Law = 8
    Media = 9
    Military = 10
    Politics = 11
    Religion = 12
    Retail = 13
    STEM = 14
    Sport = 15
    Error = 16
    '''

    '''
    ad = 0
    nonad = 1
    '''

    B_LOC = 0
    I_LOC = 1
    B_PER = 2
    I_PER = 3
    B_ORG = 4
    I_ORG = 5
    B_MISC = 6
    I_MISC = 7
    O = 8


def print_weights(weights, name):
    wr = get_csv_writer(name)
    for weight in weights:
        wr.writerow(weight)


def load_csv(path, nColSkip):
    """
    Load a CSV file.
    :param path: Path to CSV file.
    :param nColSkip: Number of initial columns to skip.
    :return: CSV file as a DataFrame.
    """
    print('Loading data...')
    data = pd.read_csv(path, header=None, low_memory=False)
    data = data.drop(data.columns[0:nColSkip], axis=1)

    assigned = data.iloc[:, len(data.columns) - 1]

    data = data.drop(len(data.columns) + nColSkip - 1, axis=1)

    data = data.as_matrix()

    dataAssigned = data, assigned

    return dataAssigned


def classify_instance(instance, weights, classes):
    """
    Classify a given instance into a class.
    :param instance: Feature vector of instance to be classified.
    :param weights: Weight vector to classify instance with.
    :param classes: List of possible classes.
    :return: Class the instance was classified as.
    """
    best = float("-inf")
    assigned = None

    for i, category in enumerate(classes):
        score = np.dot(instance, weights[i])

        if score >= best:
            best = score
            assigned = category

    return assigned


def update_weights(weights, instance, assigned, label):
    """
    Update weights according to an instance and its assigned class.
    :param weights: Initial set of weights to update.
    :param instance: Instance to update weights with.
    :param assigned: Class instance was assigned.
    :param label: Correct label for the given instance.
    :return: Updated weights.
    """
    if label != assigned:
        actual = label#.strip('.')
        #assigned = assigned.strip('.')
        weights[Classification[actual]] = np.add(weights[Classification[actual]], instance)
        weights[Classification[assigned]] = np.subtract(weights[Classification[assigned]], instance)
    return weights


def sum_weights(prevWeights, newWeights):
    """
    Perform matrix addition on two sets of weight vectors.
    :param prevWeights: List of weight vectors.
    :param newWeights:  List of weight vectors to add.
    :return: Sum of input weight vectors for each class.
    """
    for i in range(0, len(prevWeights)):
        prevWeights[i] = np.add(prevWeights[i], newWeights[i])

    return prevWeights


def divide_weights(prevWeights, n):
    """
    Divide each element in all weight vectors by n.
    :param prevWeights: List of weight vectors.
    :param n: Number to divide by.
    :return: List of weight vectors with each element divided by n.
    """
    for i in range(0, len(prevWeights)):
        prevWeights[i] = prevWeights[i]/n

    for i, weightClass in enumerate(prevWeights):
        prevWeights[i] = [int(round(j)) for j in weightClass]

    return prevWeights


def train_perceptron(weights, training, data, T, classes):
    """
    Train the perceptron on the given data.
    :param weights: Initial weights vector.
    :param training: Training data to train perceptron on.
    :param data: Pair containing ndarray of features, and a series of correct labels for each set of features.
    :param T: Number of training iterations.
    :param classes: List of possible classes.
    :return: Average weights vector for each class.
    """
    print('Training...')

    prevWeights = copy.deepcopy(weights)

    features = data[0]
    labels = data[1]
    counter = 0

    for i in range(0, T):
        for row in training:
            assigned = classify_instance(features[row], weights, classes)

            weights = update_weights(weights, features[row], assigned, labels[row])
            prevWeights = sum_weights(prevWeights, weights)
            counter += 1
            #print 'Weights updated...'

        print('Iteration complete...')

    weights = divide_weights(prevWeights, counter)

    return weights


def classify(weights, testing, data, classes):
    """
    Classify unseen instances.
    :param weights: Weights vector for each class.
    :param testing: Testing data.
    :param data: Pair containing ndarray of features, and a series of correct labels for each set of features.
    :param classes: List of possible classes.
    :return: Accuracy of weights on training data.
    """
    nCorrect = 0
    nTotal = 0

    features = data[0]
    labels = data[1]

    for row in testing:
        assigned = classify_instance(features[row], weights, classes)
        if labels[row] == assigned:
            nCorrect += 1
        nTotal += 1

    return nCorrect / nTotal


def perceptron(training, testing, data, classes, weights, T):
    """
    Train and test a perceptron on the given training and testing data.
    :param training: DataFrame of training data.
    :param testing: DataFrame of testing data.
    :param data: Pair containing ndarray of features, and a series of correct labels for each set of features.
    :param classes: List of possible classes.
    :param weights: List of weights for each class.
    :param T: Number of training iterations.
    :return: Accuracy of perceptron after training.
    """
    weights = train_perceptron(weights, training, data, T, classes)
    accuracy = classify(weights, testing, data, classes)

    print('Accuracy: %.9f' % accuracy)

    print_weights(weights, 'weights.csv')

    return accuracy


def cross_validate(nFolds, data, classes, T):
    """
    Perform k-Folds cross-validation on the perceptron.
    :param nFolds: Number of folds.
    :param data: Pair containing ndarray of features, and a series of correct labels for each set of features.
    :param classes: List of possible classes.
    :param T: Number of iterations to train perceptron on.
    """
    nAttribute = data[0].shape[1]

    sumAccuracy = 0

    kf = KFold(data[0].shape[0], nFolds, True)

    for train_index, test_index in kf:
        print('New fold...')
        weights = [[0 for x in range(nAttribute)] for x in range(len(classes))]

        sumAccuracy += perceptron(train_index, test_index, data, classes, weights, T)

    print('Average accuracy: %.9f' % (sumAccuracy / nFolds))


if __name__ == '__main__':
    """
    Main function.
    """
    path = sys.argv[1]
    # TODO What the heck is T? Oh, number of iterations. HOW IS THAT T?
    T = int(sys.argv[2])

    nColSkip = 0
    nFolds = 10

    try:
        os.remove('weights.csv')
    except OSError:
        pass

    # TODO Make a nicer way of reading classes
    classes = ['Commerce',
               'Construction',
               'Education',
               'Environment',
               'Government',
               'Health',
               'Hospitality',
               'Humanities',
               'Law',
               'Media',
               'Military',
               'Politics',
               'Religion',
               'Retail',
               'STEM',
               'Sport',
               'Error']

    data = load_csv(path, nColSkip)
    cross_validate(nFolds, data, classes, T)