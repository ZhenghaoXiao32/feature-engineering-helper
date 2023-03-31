import pandas as pd
from sklearn import metrics


def evaluate_models(clf_list, model_names, X_test, y_test):
    """
    Evaluate models in the classifier list using testing set.

    :param clf_list: the list of models/classifiers corresponding to model suffixes
    :param model_names: the list of model names
    :param X_test: testing set of X
    :param y_test: testing set of y
    :return:
    """
    test_acc = []
    f1_score = []
    for clf in clf_list:
        test_acc.append(clf.score(X_test, y_test))
        f1_score.append(metrics.f1_score(y_test, clf.predict(X_test)))
    return pd.DataFrame(data={'model': model_names, 'test_acc': test_acc, 'f1_score': f1_score})