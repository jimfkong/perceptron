import unittest
import numpy as np
import pandas as pd

from OneHot import one_hot


class OneHotTests(unittest.TestCase):
    def test_one_hot_should_generate_new_features_if_feature_has_multiple_values(self):
        features = np.array([[1, 1, 3],
                             [0, 1, 2],
                             [1, 1, 4]])

        feature_names = pd.Series(['feat1', 'feat2', 'feat3'])

        result = one_hot(features, feature_names)

        self.assertCountEqual(result[1], ['feat3_3', 'feat3_2', 'feat3_4'])

