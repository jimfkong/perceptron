import unittest
import numpy as np
import pandas as pd

from OneHot import one_hot
from Utilities import get_n_columns, get_n_rows


class OneHotTests(unittest.TestCase):
    def test_one_hot_should_not_generate_new_features_if_feature_is_binary(self):
        features = np.array([[1, 1, 1],
                            [0, 1, 1],
                            [1, 1, 0]])
        feature_names = pd.Series(['feat1', 'feat2', 'feat3'])

        result = one_hot(features, feature_names)

        self.assertEqual(result[1].tolist(), ['feat1', 'feat2', 'feat3'])
        self.assertEqual(get_n_columns(result[0]), 3)
        self.assertEqual(get_n_rows(result[0]), 3)
        self.assertEqual(result[0][0, :].tolist(), [1, 1, 1])
        self.assertEqual(result[0][1, :].tolist(), [0, 1, 1])
        self.assertEqual(result[0][2, :].tolist(), [1, 1, 0])

    def test_one_hot_should_generate_new_features_if_feature_has_multiple_values(self):
        features = np.array([[1, 1, 3],
                             [0, 1, 2],
                             [1, 1, 4]])

        feature_names = pd.Series(['feat1', 'feat2', 'feat3'])

        result = one_hot(features, feature_names)

        self.assertEqual(result[1].tolist(), ['feat1', 'feat2', 'feat3_2', 'feat3_3', 'feat3_4'])
        self.assertEqual(get_n_columns(result[0]), 5)
        self.assertEqual(get_n_rows(result[0]), 3)
        self.assertEqual(result[0][0, :].tolist(), [1, 1, 0, 1, 0])
        self.assertEqual(result[0][1, :].tolist(), [0, 1, 1, 0, 0])
        self.assertEqual(result[0][2, :].tolist(), [1, 1, 0, 0, 1])
