import pandas as pd

def one_hot(features, feature_names):
    # Would need to:
    #     Get the current features
    #     for each feature, check if it has multiple possible values
    #     if it has multiple values, then create a new feature for each found value

    # if replacing a feature: need to get rid of the existing feature
    # then expand the feature matrix with new columns

    one_hot_features = pd.DataFrame()
    features_to_remove = []

    for column in range(0, len(features.columns)):
        unique_column_values = get_unique_values_in_column(features, column)

        if len(unique_column_values) > 2:
            features_to_remove.append(column)

            for value in unique_column_values:
                one_hot_feature_name = feature_names[column] + "_is_" + value
                one_hot_features[one_hot_feature_name] = 0

                for row in range(0, len(features.rows)):
                    one_hot_features.iloc[row, one_hot_feature_name] = features.iloc[row, column] == value
                # add a new column with df[idx] = defaultvalue
                # Need to then parse each row to generate the feature


def get_unique_values_in_column(data_frame, column_idx):
    return data_frame.iloc[: column_idx].unique()