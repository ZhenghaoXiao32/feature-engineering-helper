"""Frequency Encoding

This function performs the frequency encoding for training and testing set.
"""


def freq_encoder(train_set, test_set, column_list, suffix='_freq'):
    """An inplace method to encode categorical columns using category frequencies.

    :param train_set: the training set to be encoded
    :param test_set: the testing set to be encoded
    :param column_list: the list of columns to be frequency encoded
    :param suffix: the suffix of the encoded new column
    :return: No returns since it is an inplace method
    """
    for col in column_list:
        # create frequency encoder
        _freq_encoder = train_set.groupby(col).size() / len(train_set)
        # fit_transform for train and test set
        train_set[col + suffix] = train_set[col].apply(lambda x: _freq_encoder[x])
        test_set[col + suffix] = test_set[col].apply(lambda x: _freq_encoder[x])