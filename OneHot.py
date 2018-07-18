import pandas as pd
import numpy as np


from Utilities import get_n_columns, get_n_rows


def one_hot(features, feature_names):
    # Would need to:
    #     Get the current features
    #     for each feature, check if it has multiple possible values
    #     if it has multiple values, then create a new feature for each found value

    # if replacing a feature: need to get rid of the existing feature
    # then expand the feature matrix with new columns

    one_hot_features = np.zeros((get_n_rows(features), 0), int)
    one_hot_feature_names = []
    features_to_remove = []

    for column in range(0, get_n_columns(features)):
        unique_column_values = get_unique_values_in_column(features, column)

        if len(unique_column_values) > 2:
            features_to_remove.append(column)

            for value in unique_column_values:
                one_hot_feature = np.zeros((get_n_rows(features), 1), int)
                one_hot_feature_names.append(feature_names[column] + "_is_" + value)

                for row in range(0, get_n_rows(features)):
                    one_hot_feature[row] = features[row, column] == value

                np.vstack(one_hot_feature)
                one_hot_features = np.hstack((one_hot_features, one_hot_feature))

    return None


def get_unique_values_in_column(ndarray, column_idx):
    return np.unique(ndarray[:, column_idx])
