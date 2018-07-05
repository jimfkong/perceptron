from __future__ import division

import re
import string
import unicodedata
from collections import Counter

import Feature

from ConllTags import *
from nltk import PorterStemmer


def load_gazette(file, capitalise):
    """
    Load a gazette from a file into a list.
    :param file: File to load gazette from.
    :param capitalise: True to convert words in gazette to title-case, false otherwise.
    :return: List of words contained in gazette.
    """
    names = []

    with open(file, 'r') as f:
        for line in f:
            if not capitalise:
                names.append(line.strip())
            else:
                names.append(line.capitalize().strip())

    return names


def manage_word_flags(word, additional_feats):
    """
    Process word-flags for a given word.
    :param word: Word to process word-flags over.
    :param additional_feats: Counter of features to add word-flags to.
    :return: Counter of features with word-flags added.
    """
    if word.isdigit():
        additional_feats['digits'] += 1
    if word.islower():
        additional_feats['lower'] += 1
    if word.isupper():
        additional_feats['upper'] += 1

    if word.istitle():
        additional_feats['title'] += 1

    if '.' in word:
        additional_feats['period'] += 1
    pattern = re.compile("[\d{}]+$".format(re.escape(string.punctuation)))
    if (pattern.match(word)):
        additional_feats['punct'] += 1

    return additional_feats


def replace_weird_char(word):
    """
    Replace characters in a string that can't be in variable names.
    :param word: Word containg characters to replace.
    :return: Word with characters replaced.
    """
    word = word.replace('-', '_')
    word = word.replace('$', 'dollar')
    word = word.replace('.', 'dot')
    word = word.replace('"', 'quotation')
    word = word.replace(',', 'comma')
    word = word.replace('(', 'o_bracket')
    word = word.replace(')', 'c_bracket')
    word = word.replace(':', 'colon')
    word = word.replace('\'\'', 'apostrophe')

    return word


def process_file(file):
    """
    Extract features from each line in the input file.
    :param file: File to extract features from.
    :return: List of sentences, where each sentence is composed of Features representing the words in
    the sentence.
    """
    sentences = []
    sentence = []
    names_gazette = load_gazette('Given-Names', True)
    place_gazette = load_gazette('places', True)
    surnames_gazette = load_gazette('surnames', True)
    portStem = PorterStemmer()

    with open(file, 'r') as f:
        for line in f:
            if line.startswith('-DOCSTART-'):
               continue
            if line.isspace():
                if sentence: # If sentence is not empty
                    sentences.append(sentence)
                sentence = []
                continue

            line = line.strip() # Remove trailing whitespace
            tokens = line.split(' ')
            for i in range(1, len(tokens)):
                tokens[i] = replace_weird_char(tokens[i])

            pos_tag = [0] * len(PoSTag.__members__)
            chunk = [0] * len(SyntacticChunk.__members__)

            p_tag = tokens[1]
            c_tag = tokens[2]
            g_standard = tokens[3]

            pos_tag[PoSTag[p_tag]] = 1
            chunk[SyntacticChunk[c_tag]] = 1

            additional_feats = Counter()
            word_stem = None

            word_stem = unicodedata.normalize('NFKD', portStem.stem(tokens[0].lower())).encode('ascii', 'ignore')

            if tokens[0] in names_gazette:
                additional_feats['named'] += 1

            if tokens[0] in place_gazette:
                additional_feats['place'] += 1

            if tokens[0] in surnames_gazette:
                additional_feats['surname'] += 1

            additional_feats = manage_word_flags(tokens[0], additional_feats)

            token_feat = Feature.Feature(tokens[0], PoSTag[p_tag], SyntacticChunk[c_tag],
                                         g_standard, additional_feats, word_stem)

            if sentence:
                token_feat.set_prev(sentence[-1])
                sentence[-1].set_next(token_feat)

            sentence.append(token_feat)

    return sentences
