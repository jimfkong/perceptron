import pandas as pd
import numpy as np

from Utilities import get_n_columns, get_n_rows, drop_column


def one_hot(features, feature_names):
    one_hot_features = np.zeros((get_n_rows(features), 0), int)
    one_hot_feature_names = []
    features_to_remove = []

    for column in range(0, get_n_columns(features)):
        unique_column_values = get_unique_values_in_column(features, column)

        if len(unique_column_values) > 2:
            features_to_remove.append(column)

            for value in unique_column_values:
                one_hot_feature = np.zeros((get_n_rows(features), 1), int)
                one_hot_feature_names.append(feature_names[column] + "_" + str(value))

                for row in range(0, get_n_rows(features)):
                    one_hot_feature[row] = features[row, column] == value

                np.vstack(one_hot_feature)
                one_hot_features = np.hstack((one_hot_features, one_hot_feature))

    for to_remove in features_to_remove:
        features = drop_column(features, to_remove)
        del(feature_names[to_remove])

    one_hot_features = np.append(features, one_hot_features, 1)
    one_hot_feature_names = feature_names.append(pd.Series(one_hot_feature_names))

    return one_hot_features, one_hot_feature_names


def get_unique_values_in_column(ndarray, column_idx):
    return np.unique(ndarray[:, column_idx])
