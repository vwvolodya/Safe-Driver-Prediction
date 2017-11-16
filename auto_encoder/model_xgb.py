import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import auc, precision_score, recall_score, roc_curve
from sklearn.model_selection import GridSearchCV
from random import choice
from tqdm import tqdm as progressbar
from pprint import pprint

from sklearn.externals import joblib


def Gini(y_true, y_pred):
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]

    # sort rows on prediction column
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:, 0].argsort()][::-1, 0]
    pred_order = arr[arr[:, 1].argsort()][::-1, 0]

    # get Lorenz curves
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(1 / n_samples, 1, n_samples)

    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)

    # normalize to true Gini coefficient
    return G_pred / G_true


def load_data():
    X = np.load("X.npy")
    Y = np.load("Y.npy")
    print(X.shape)
    print(Y.shape)
    test_x = np.load("X_test.npy")
    test_y = np.load("Y_test.npy")
    print("loaded")
    return X, Y, test_x, test_y


def evaluate(model, data, y_true, verbose=True):
    y_pred = model.predict(data)
    y_prob = model.predict_proba(data)
    y_prob = y_prob[:, 1]
    if verbose:
        print(y_pred.shape)
        print(y_prob.shape)

        print(precision_score(y_true, y_pred))
        print(recall_score(y_true, y_pred))

    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1.0)
    auc_metric = auc(fpr, tpr)
    custom_gini = 2 * auc_metric - 1
    # gin = Gini(y_true, y_prob)

    if verbose:
        print(custom_gini)
        print(auc_metric)
        # print(gin)
    return auc_metric, custom_gini


def searcher(params, best_score, x, y, test_x, test_y, iterations=20):
    visited = set()
    for i in progressbar(range(iterations)):
        sample = {k: choice(v) for k, v in params.items()}
        identifier = hash(tuple(list(sample.items())))
        if identifier in visited:
            continue
        visited.add(identifier)
        mdl = XGBClassifier(**sample)
        mdl.fit(x, y)
        a, cg = evaluate(mdl, test_x, test_y)
        print(a, cg)
        if cg > best_score:
            print("Found new better params")
            pprint(mdl.get_params())
            print("New best score is", cg)
            best_score = cg
            joblib.dump(mdl, "xg.mdl")


if __name__ == "__main__":
    X, Y, test_X, test_Y = load_data()
    base_level = 0.273
    param_grid = {'base_score':         [0.5],
                  'colsample_bylevel':  [0.9, 0.95],
                  "colsample_bytree":   [0.9, 0.95],
                  'gamma':              [0.1],
                  'max_delta_step':     [1],
                  'max_depth':          [5, 4, 3],
                  'min_child_weight':   [1],
                  'n_estimators':       [110, 120, 130],
                  'reg_alpha':          [0.7, 0.75, 0.8],
                  'subsample':          [0.9, 0.95],
                  'scale_pos_weight':   [1, 2, 3, 4],
                  'reg_lambda':         [0.9, 0.95],
                  'learning_rate': [0.06, 0.065]}

    searcher(param_grid, base_level, X, Y, test_X, test_Y, iterations=200)

    # mdl_xg = XGBClassifier(max_depth=5, learning_rate=0.05, reg_lambda=0.5, seed=101, scale_pos_weight=5)
    # print(mdl_xg.get_params())
    # # exit(1)
    # mdl_xg.fit(X, Y, eval_metric="auc")
    # print("Fit")
    # evaluate(mdl_xg, test_X, test_Y)
    # joblib.dump(mdl_xg, "xg.mdl")
