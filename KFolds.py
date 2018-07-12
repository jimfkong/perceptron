import random
import math
from collections import namedtuple

Fold = namedtuple('Fold', ['train_index', 'test_index'])


def kfolds(n_rows, n_folds, seed=None):
    random.seed(seed)
    shuffled = random.sample(range(0, n_rows), n_rows)

    fold_size = math.floor(n_rows / n_folds)

    folds = []

    for i in range(0, n_folds):
        start_idx = fold_size * i
        end_idx = start_idx + fold_size
        # TODO Test if we're skipping indexes by accident, or if we missed any (e.g. remainder due to fold sizes,
        # or index out of bounds)
        # TODO Do we need a deepcopy?
        train_index = shuffled[start_idx:end_idx]
        test_index = shuffled[0:start_idx] + shuffled[end_idx:]

        folds.append(Fold(train_index, test_index))

    return folds

