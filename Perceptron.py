import copy
import numpy as np


class Perceptron:
    weights = []
    classes = []

    def __init__(self, n_attributes, classes):
        self.weights = [[0 for x in range(n_attributes)] for x in range(len(classes))]
        self.classes = classes

    def train(self, training, data, n_iterations):
        print('Training...')

        prev_weights = copy.deepcopy(self.weights)

        features = data[0]
        labels = data[1]
        counter = 0

        for i in range(0, n_iterations):
            for row in training:
                assigned = self.classify_instance(features[row])

                self.__update_weights(features[row], assigned, labels[row])
                prev_weights = self.__sum_weights(prev_weights, self.weights)
                counter += 1
                # print 'Weights updated...'

            print('Iteration complete...')

        self.weights = self.__divide_weights(prev_weights, counter)

    def classify_instance(self, instance):
        """
        Classify a given instance into a class.
        :param instance: Feature vector of instance to be classified.
        :return: Class the instance was classified as.
        """
        best = float("-inf")
        assigned = None

        for i, category in enumerate(self.classes):
            score = np.dot(instance, self.weights[i])

            if score >= best:
                best = score
                assigned = category

        return assigned

    def __update_weights(self, instance, assigned, label):
        """
        Update weights according to an instance and its assigned class.
        :param instance: Instance to update weights with.
        :param assigned: Class instance was assigned.
        :param label: Correct label for the given instance.
        :return: Updated weights.
        """
        if label != assigned:
            actual = label

            self.weights[self.classes.index(actual)] = np.add(self.weights[self.classes.index(actual)], instance)
            self.weights[self.classes.index(assigned)] = np.subtract(self.weights[self.classes.index(assigned)], instance)

    def __sum_weights(self, prev_weights, new_weights):
        """
        Perform matrix addition on two sets of weight vectors.
        :param prev_weights: List of weight vectors.
        :param new_weights:  List of weight vectors to add.
        :return: Sum of input weight vectors for each class.
        """
        for i in range(0, len(prev_weights)):
            # TODO Is this the correct logic? Am I modifying the weight vectors correctly?
            prev_weights[i] = np.add(prev_weights[i], new_weights[i])

        return prev_weights

    def __divide_weights(self, prev_weights, n):
        """
        Divide each element in all weight vectors by n.
        :param prev_weights: List of weight vectors.
        :param n: Number to divide by.
        :return: List of weight vectors with each element divided by n.
        """
        for i in range(0, len(prev_weights)):
            prev_weights[i] = prev_weights[i] / n

        for i, weightClass in enumerate(prev_weights):
            prev_weights[i] = [int(round(j)) for j in weightClass]

        return prev_weights
