import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

RANDOM_SEED = 2022


def train_models(X, y, model_suffix, clf_list, params_list, method='random'):
    """
    This function train multiple models using cross validation randomized search or grid search,
    then store the validation result and the best models.

    :param X: training set of X
    :param y: training set of y
    :param model_suffix: the list of suffixes for each model
    :param clf_list: the list of models/classifiers corresponding to model suffixes
    :param params_list: the list of hyperparameter searching spaces for models in the list
    :param method: hyperparameter search method: "random" or "grid", notice the "random" method
                   requires a RANDOM_SEED to be set in the global environment
    :return:
    """
    if method == 'random':
        for i in range(len(clf_list)):
            # model training with RandomizedSearchCV
            rscv = RandomizedSearchCV(estimator=clf_list[i],
                                      param_distributions=params_list[i],
                                      n_jobs=-1, random_state=RANDOM_SEED).fit(X, y)
            # store cv results
            globals()['rscv%s' % model_suffix[i]] = pd.DataFrame(rscv.cv_results_)
            # store the best model
            globals()['best%s' % model_suffix[i]] = rscv
            print("rscv", model_suffix[i], " is trained", sep='')
    if method == 'grid':
        for i in range(len(clf_list)):
            # model training with GridSearchCV
            gscv = GridSearchCV(estimator=clf_list[i],
                                param_grid=params_list[i],
                                cv=5).fit(X, y)
            # store cv results
            globals()['gscv%s' % model_suffix[i]] = pd.DataFrame(gscv.cv_results_)
            # store the best model
            globals()['best%s' % model_suffix[i]] = gscv
            print("gscv", model_suffix[i], " is trained", sep='')
