from __future__ import division
from collections import Counter

from enum import IntEnum

import numpy as np
from ConllTags import NERTag


class Feat(IntEnum):
    """
    Enum of current features.
    """
    POS = 0
    CHUNK = 1
    PREV_POS = 2
    PREV_PREV_POS = 3
    NEXT_POS = 4
    NEXT_NEXT_POS = 5
    BOW = 6
    BOW_PREV = 7
    BOW_PREV_PREV = 8
    BOW_NEXT = 9
    BOW_NEXT_NEXT = 10
    STEM = 11
    '''
    BOW = 0
    POS = 1
    CHUNK = 2
    '''

class Weight():
    """
    Weight vector for a given NER tag.
    """
    def __init__(self, n_tag):
        """
        Create a new weight vector.
        :param n_tag: Number of NER tags.
        """
        self.weights = []
        for vector in Feat:
            self.weights.append(Counter())

        self.additional_feats = Counter()
        self.prev_prediction = np.zeros(n_tag + 1)

    def dot_prev_prediction(self, assumed_prev):
        """
        Dot product of only the previous prediction vectors.
        :param assumed_prev: Assumed previous NER tag.
        :return: The current weight value of the assumed previous prediction.
        """
        if assumed_prev is not None:
            return self.prev_prediction[assumed_prev]
        else:
            return self.prev_prediction[-1]

    def dot(self, other, assumed_prev_tag, exclude_prev):
        """
        Perform a dot product between the stored weights and a given Feature.
        :param other: Feature to dot product.
        :param assumed_prev_tag: The assumed NER tag of the previous word.
        :param exclude_prev: True to exclude the previous prediction feature.
        :return: Dot product of the stored weights and the given Feature.
        """
        result = 0.0

        for vector in Feat:
            if other.values[vector] is not None:
                result += self.weights[vector][other.values[vector]]

        if not exclude_prev:
            if assumed_prev_tag is not None:
                result += self.prev_prediction[assumed_prev_tag]
            else:
                result += self.prev_prediction[-1]

        result += sum((self.additional_feats[key] * other.additional_feats[key]) for key in other.additional_feats)

        return result

    def subtract(self, other, assumed_prev_tag):
        """
        Subtract a given Feature from the stored weights.
        :param other: Feature to subtract from the stored weights.
        :param assumed_prev_tag: The assumed NER tag of the previous word.
        """
        for vector in Feat:
            if other.values[vector] is not None:
                self.weights[vector][other.values[vector]] -= 1

        if assumed_prev_tag is not None:
            self.prev_prediction[assumed_prev_tag] -= 1
        else:
            self.prev_prediction[-1] -= 1

        for key in other.additional_feats:
            self.additional_feats[key] -= other.additional_feats[key]

    def add(self, other, assumed_prev_tag):
        """
        Add a given Feature to the stored weights.
        :param other: Feature to add to the stored weights.
        :param assumed_prev_tag: The assumed NER tag of the previous word.
        """
        for vector in Feat:
            if other.values[vector] is not None:
                self.weights[vector][other.values[vector]] += 1

        if assumed_prev_tag is not None:
            self.prev_prediction[assumed_prev_tag] += 1
        else:
            self.prev_prediction[-1] += 1

        for key in other.additional_feats:
            self.additional_feats[key] += other.additional_feats[key]

    def combine(self, other):
        """
        Combine the current weight vector with another weight vector.
        :param other: Other weight vector to merge in.
        """
        for vector in Feat:
            self.weights[vector].update(other.weights[vector])

        self.additional_feats.update(other.additional_feats)

        for i in range(0, len(NERTag.__members__) + 1):
            self.prev_prediction[i] += other.prev_prediction[i]

    def divide(self, divisor):
        """
        Divide all weights in the weight vector by a given divisor.
        :param divisor: Value to divide all weights by.
        """
        for vector in Feat:
            self.weights[vector].update((word, int(round(count / divisor))) for word, count in self.weights[vector].items())

        self.additional_feats.update((key, int(round(count / divisor))) for key, count in self.additional_feats.items())

        for i in range(0, len(NERTag.__members__) + 1):
            self.prev_prediction[i] = int(round(self.prev_prediction[i] / divisor))

    def print_weights(self, threshold):
        """
        Print the current weight values, if higher than a given threshold.
        :param threshold: Minimum weight value to print.
        """
        for vector in Feat:
            print(vector.name)
            for word, count in self.weights[vector].iteritems():
                if abs(count) > threshold:
                    print('{0}\t{1}'.format(word, count))
            print('')

        print('History: ')
        for tag in NERTag:
            print(str(tag.name) + ' ' + str(self.prev_prediction[tag]),)
        print('\n')

        print('Additional features:')
        for word, count in self.additional_feats.iteritems():
            if abs(count) > threshold:
                print('{0}\t{1}'.format(word, count))
        print('\n')
