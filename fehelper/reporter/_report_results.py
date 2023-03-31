import pandas as pd


def report_results(model_list, model_suffix):
    """
    Store the best validation results into a dataframe.

    :param model_list: the list of models
    :param model_suffix: the list of model suffixes
    :return:
    """
    for i in range(len(model_list)):
        globals()['df%s' % model_suffix[i]] = model_list[i].query('rank_test_score == 1')\
        [['params', 'mean_test_score', 'std_test_score']]