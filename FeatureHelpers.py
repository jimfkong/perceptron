from __future__ import division
import codecs
import csv
import json
import os
import nltk
import numpy as np
import unicodedata
from string import digits, punctuation
from nltk import PorterStemmer


def get_tokens(file):
    """
    Retrieve tokens from a given file.
    :param file: File to retrieve tokens from.
    :return: List of tokens extracted from file.
    """
    fileContent = codecs.open(file, 'r', encoding='utf8').read()
    fileContent = unicodedata.normalize('NFKD', fileContent).encode('ascii', 'ignore')
    fileContent = fileContent.translate(None, digits + punctuation)


    return nltk.word_tokenize(fileContent)

def stem_words(words):
    """
    Stem list of given words.
    :param words: List of words to stem.
    :return: Set of unique stemmed words.
    """
    out = set()
    portStem = PorterStemmer()

    for word in words:
        out.add(unicodedata.normalize('NFKD', portStem.stem(word)).encode('ascii', 'ignore'))

    return out


def read_json(file):
    """
    Read a JSON file.
    :param file: Name of input file.
    :return: JSON data extracted from input file.
    """
    with open(file) as data_file:
        data = json.load(data_file)

    return data


def set_to_lower(input):
    """
    Convert input to a lowercase set.
    :param input: List-like structure to convert.
    :return: Input converted to lowercase set.
    """
    return set(map(lambda x: x.lower(), input))


def get_csv_writer(output):
    """
    Get a new CSV writer object.
    :param output: Name of the CSV output file.
    :return: CSV writer for given output file.
    """
    out = open(output, "ab")
    wr = csv.writer(out, dialect='excel')

    return wr


def get_actual_class(file, labels):
    """
    Retrieve the actual class assigned to a given file.
    :param file: File to retrieve actual class for.
    :param labels: Location of CSV file containing files and their associated label.
    :return: Class associated with given file.
    """
    labels = np.genfromtxt(labels, delimiter=',', dtype=None)

    basename = os.path.splitext(os.path.basename(file))[0]
    row, column = np.where(labels == basename)
    lastIndex = labels.shape[1]

    assigned = labels[row, lastIndex - 1]

    return assigned[0]


def decode_unicode(words):
    """
    Decode a given list of words from unicode to ascii.
    :param words: List of words to decode.
    :return: List of words, encoded as ascii.
    """
    words = [unicodedata.normalize('NFKD', word).encode('ascii', 'ignore') for word in words]

    return words