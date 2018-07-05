from __future__ import division

import copy
import os
import sys
import Weight
import FeatureGenerator

from ConllTags import *
from Weight import Feat
from FeatureHelpers import get_csv_writer

THRESHOLD = 30000


def get_best_viterbi_score(word_i, word_feats, current_tag, v_table, weights):
    """
    Retrieve the highest viterbi score for a given word and NER tag.
    :param word_i: The current words index in its sentence.
    :param word_feats: Context of the current word.
    :param current_tag: The NER tag being considered.
    :param v_table: 2D array representing the viterbi table of previous scores.
    :param weights: List containing current weights for each NER tag.
    :return: The best viterbi score for the current tag, and the previous tag the viterbi score
    was generated from.
    """
    best = 0.0
    best_tag = None
    score = 0.0

    base_score = weights[current_tag].dot(word_feats, None, True)

    for tag in NERTag:
        if word_i > 0:
            score = v_table[word_i - 1][tag]
        score += base_score
        score += weights[current_tag].dot_prev_prediction(tag)

        if score >= best:
            best = score
            best_tag = tag

    return best, best_tag


def retrieve_viterbi_sequence(v_table, tag_table):
    """
    Retrieve the best sequence of NER tags.
    :param v_table: 2D array representing the viterbi table.
    :param tag_table: 2D array representing the tag history.
    :return: List containing the best sequence of NER tags.
    """
    predicted_tags = []
    best = float('-inf')
    best_tag = None
    n_tokens = len(v_table) - 1


    for tag in NERTag:
        if v_table[n_tokens][tag] >= best:
            best = v_table[n_tokens][tag]
            best_tag = tag

    if best_tag is not None:
        predicted_tags.append(best_tag)

        for i, j in zip(range(n_tokens, 0, -1), range(0, n_tokens)):
            predicted_tags.append(tag_table[i][predicted_tags[j]])

        predicted_tags.reverse()

    return predicted_tags


def update_weights(weights, instance, assigned, prev_assigned):
    """
    Update the weights according to an instance and its assigned NER tag.
    :param weights: List of current weights.
    :param instance: Instance to update weights with.
    :param assigned: NER tag assigned to instance.
    :param prev_assigned: The NER tag assigned to the previous word.
    """
    if instance.gold_standard != assigned.name:
        weights[NERTag[instance.gold_standard]].add(instance, prev_assigned)
        weights[assigned].subtract(instance, prev_assigned)


def viterbi(sentence, weights):
    """
    Perform viterbi over a sentence to retrieve the best NER tag sequence.
    :param sentence: List containing features representing each word in the sentence.
    :param weights: List of weight vectors.
    :return: List containing the best sequence of NER tags for the given sentence.
    """
    v_table = [[0 for x in range(len(NERTag.__members__))] for y in range(len(sentence))]
    tag_table = [[0 for x in range(len(NERTag.__members__))] for y in range(len(sentence))]

    for word_i, word in enumerate(sentence):
        for tag in NERTag:
            #If first token, then set score proportional to probability that token is tag only.
            if word_i < 1:
                v_table[word_i][tag] = weights[tag].dot(word, None, False)
            else:
                v_table[word_i][tag], tag_table[word_i][tag] = \
                    get_best_viterbi_score(word_i, word, tag, v_table, weights)

    return retrieve_viterbi_sequence(v_table, tag_table)


def reverse_replacements(word):
    """
    Reverse the replacement of characters that can't be in variable names.
    :param word: Word to replace characters with.
    :return: Word with relevant characters replaced.
    """
    word = word.replace('_', '-')
    word = word.replace('dollar', '$')

    word = word.replace('dot', '.')
    word = word.replace('quotation', '"')
    word = word.replace('comma', ',')
    word = word.replace('o_bracket', '(')
    word = word.replace('c_bracket', ')')
    word = word.replace('colon', ':')
    word = word.replace('apostrophe', '\'\'')

    return word


def print_sentence_output(sentence, prediction, wr):
    """
    Print each word in a sentence, along with its POS tag, syntactic chunk, gold-standard, and assigned NER tag.
    :param sentence: List of words to print.
    :param prediction: List of predicted tags for each word.
    :param wr: File writer to write to file with.
    """
    prediction = [reverse_replacements(word.name) for word in prediction]

    for i, word in enumerate(sentence):
        tokens = []
        tokens.append(word.values[Feat.BOW])
        tokens.append(word.values[Feat.POS].name)
        tokens.append(word.values[Feat.CHUNK].name)
        tokens.append(word.gold_standard)

        for j in range(1, len(tokens)):
            tokens[j] = reverse_replacements(tokens[j])
        for token in tokens:
            wr.write(token + ' ')
        wr.write(prediction[i] + '\n')
    wr.write('\n')


def train_viterbi_avg_perceptron(weights, T, sentences):
    """
    Train an averaged perceptron that incorporates sequence modelling via the Viterbi algorithm.
    :param weights: List of weights for each NER tag.
    :param T: Number of iterations to train the perceptron.
    :param sentences: List of sentences to train the perceptron on.
    :return: Averaged weight vector.
    """
    prev_weights = copy.deepcopy(weights)
    counter = 0

    for t in range(0, T):
        print('New iteration...')
        for sentence in sentences:
            sequence = viterbi(sentence, weights)
            for i, tag in enumerate(sequence):
                prev_tag = None
                if i > 0:
                    prev_tag = sequence[i - 1]
                update_weights(weights, sentence[i], tag, prev_tag)

            for i in range(0, len(prev_weights)):
                prev_weights[i].combine(weights[i])
            counter += 1

    print('Averaging weights...')
    for i in range(0, len(prev_weights)):
        prev_weights[i].divide(counter)

    return prev_weights


def perceptron_classify(word, weights):
    """
    Classify a word based on the given weights.
    :param word: Word to classify.
    :param weights: List of weights for each possible tag.
    :return: Best tag assigned to word.
    """
    best = float("-inf")
    assigned = None

    for tag in NERTag:
        score = weights[tag].dot(word, None, True)

        if score >= best:
            best = score
            assigned = tag

    return assigned


def train_perceptron(weights, T, sentences):
    """
    Train an averaged perceptron without the Viterbi algorithm.
    :param weights: List of weights for each possible tag.
    :param T: Number of iterations to train the perceptron.
    :param sentences: List of sentences to train the perceptron on.
    :return: Averaged weight vector.
    """
    prev_weights = copy.deepcopy(weights)
    counter = 0

    for i in range(T):
        for sentence in sentences:
            for word in sentence:
                assigned = perceptron_classify(word, weights)
                update_weights(weights, word, assigned, None)
            for i in range(0, len(prev_weights)):
                prev_weights[i].combine(weights[i])
            counter += 1

    print('Averaging weights...')
    for i in range(0, len(prev_weights)):
        prev_weights[i].divide(counter)

    return prev_weights


if __name__ == '__main__':
    train = sys.argv[1]
    test = sys.argv[2]
    out = sys.argv[3]

    T = 3

    print('Extracting training features...')
    sentences = FeatureGenerator.process_file(train)
    weights = []

    n_additional_feats = len(sentences[0][0].additional_feats)

    print('Training...')
    for tag in range(0, len(NERTag.__members__)):
        new_weight = Weight.Weight(len(NERTag.__members__))
        weights.append(new_weight)

    weights = train_viterbi_avg_perceptron(weights, T, sentences)
    #weights = train_perceptron(weights, T, sentences)

    accuracy = 0
    n_sentences = 0

    print('Extracting testing features...')
    sentences = FeatureGenerator.process_file(test)

    try:
        os.remove(out)
    except OSError:
        pass
    try:
        os.remove('confuse.csv')
    except OSError:
        pass

    wr = open(out, 'wb')
    csv_wr = get_csv_writer('confuse.csv')

    print('Testing...')
    confuse = [[0 for x in range(len(NERTag.__members__))] for x in range(len(NERTag.__members__))]

    '''
    for sentence in sentences:
        sequence = []
        for word in sentence:
            assigned = perceptron_classify(word, weights)
            sequence.append(assigned)
            confuse[assigned][NERTag[word.gold_standard]] += 1

        print_sentence_output(sentence, sequence, wr)
    '''

    for sentence in sentences:
        correct = 0
        total = 0
        sequence = viterbi(sentence, weights)
        for i, tag in enumerate(sequence):
            if tag.name == sentence[i].gold_standard:
                correct += 1
            confuse[tag][NERTag[sentence[i].gold_standard]] += 1
            total += 1

        accuracy += (correct / total)
        n_sentences += 1

        print_sentence_output(sentence, sequence, wr)

    accuracy /= n_sentences

    csv_wr.writerows(confuse)
    wr.close()
    print('Average accuracy: ' + str(accuracy))

    for tag in NERTag:
        print('Weights for ' + str(tag.name) + ': ')
        weights[tag].print_weights(THRESHOLD)
