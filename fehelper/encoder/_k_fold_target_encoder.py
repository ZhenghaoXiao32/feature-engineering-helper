import numpy as np
import pandas as pd
from sklearn import base
from sklearn.model_selection import KFold


class KFoldTargetEncoderTrain(base.BaseEstimator, base.TransformerMixin):
    """
    This object contains a target encoder for a training set which should have
    both X and y.

    Arguments:
    ---------
    feature:          string. Name of the feature in the training set.

    target:           string. Name of the target in the training set.

    n_fold:           default 5. Number of folds to use in KFold.

    verbose:          bool, default True. If set to True, the correlation between the
                      feature and the target will be calculated and printed out.

    discard_original: bool,, default False. If set to True, the feature column will be
                      deleted from the training set.

    Example:
    ---------
    train_target_encoder = KFoldTargetEncoderTrain(feature='A', target='target')

    new_train = train_target_encoder.fit_transform(train)
    """

    def __init__(self, feature, target, n_fold=5, verbose=True, discard_original=False):

        self.feature = feature
        self.target = target
        self.n_fold = n_fold
        self.verbose = verbose
        self.discard_original = discard_original

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform the original training set. Notice this function can only encode
        one feature once.

        Arguments:
        ----------
        X: A pandas DataFrame which should include both the feature and the target.

        Output:
        -------
        X: A pandas DataFrame with the target encoding.
        """

        # notice this function can only encode one feature at a time
        assert (type(self.feature) == str)
        assert (type(self.target) == str)
        assert (self.feature in X.columns)
        assert (self.target in X.columns)

        mean_of_target = X[self.target].mean()
        kf = KFold(n_splits=self.n_fold, shuffle=True, random_state=42)
        # create the target encoding
        col_mean_name = self.feature + '_target'
        X[col_mean_name] = np.nan

        for train_index, val_index in kf.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            X.loc[X.index[val_index], col_mean_name] = \
                X_val[self.feature].map(X_train.groupby(self.feature)[self.target].mean())
        # missing value imputation
        X[col_mean_name].fillna(mean_of_target, inplace=True)

        if self.verbose:
            encoded_feature = X[col_mean_name].values
            print('Correlation between {} and {} is {}.'. \
                  format(col_mean_name, self.target,
                         np.corrcoef(X[self.target].values, encoded_feature)[0][1]))
        # discard original feature column if needed
        if self.discard_original:
            X = X.drop(self.target, axis=1)

        return X


class KFoldTargetEncoderTest(base.BaseEstimator, base.TransformerMixin):
    """
    This object contains a target encoder for a testing set which should have
    both X and y.

    Arguments:
    ---------
    train:          pandas DataFrame. The training DataFrame with the feature and
                    the target encoded column of it.

    feature:        string. The column name of the feature.

    feature_target: string. The column name of the feature_target that
                    has been calculated in the training set.

    Example:
    ---------
    test_target_encoder = KFoldTargetEncoderTest(new_train, 'A', 'A_target')

    new_test = test_target_encoder.transform(test)
    """

    def __init__(self, train, feature, feature_target):
        self.train = train
        self.feature = feature
        self.feature_target = feature_target

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform the testing set based on K-fold target encoder of the training set.
        Notice this function can only encode one feature at a time.

        Argument
        --------
        X: pandas DataFrame. The testing set to be transformed.

        Output
        --------
        X: A pandas DataFrame with transformed target encoding.
        """

        mean = self.train[[self.feature, self.feature_target]].groupby(self.feature).mean().reset_index()

        dd = {}
        for index, row in mean.iterrows():
            dd[row[self.feature]] = row[self.feature_target]

        X[self.feature_target] = X[self.feature]
        X = X.replace({self.feature_target: dd})

        return X


def k_fold_target_encoder(x_train, y_train, x_test, y_test, column_list, suffix='_target', n_fold=5):
    """A function to perform k-fold target encoding.

    :param x_train: the training set contains only the independent variables
    :param y_train: the training set contains only the dependent variables
    :param x_test: the testing set contains only the independent variables
    :param y_test: the testing set contains only the dependent variables
    :param column_list: the list of columns to be encoded
    :param suffix: the suffix of the encoded new column
    :return: the encoded training and testing set along with the correlation info between the encoded feature and the
     target
    """
    # fit-transform the training data
    train_set = pd.concat([x_train, y_train], axis=1)
    for col in column_list:
        _train_target_encoder = KFoldTargetEncoderTrain(col, 'Y', n_fold=n_fold)
        train_set = _train_target_encoder.fit_transform(train_set)
    # transform the testing data
    test_set = pd.concat([x_test, y_test], axis=1)
    target_column_list = []
    for col in column_list:
        target_column_list.append(col + suffix)
    for col, target_col in zip(column_list, target_column_list):
        _test_target_encoder = KFoldTargetEncoderTest(train_set, col, target_col)
        test_set = _test_target_encoder.transform(test_set)

    return train_set, test_set
