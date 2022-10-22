import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score


def auc_score(y, y_pred):
    return roc_auc_score(np.eye(y_pred.shape[1])[y], y_pred, average="macro")


def lr_mean_auc_score(X, y):
    kfold = KFold(shuffle=True)
    aucs = []
    for train, test in kfold.split(X):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict_proba(X_test)
        aucs.append(auc_score(y_test, y_pred))
    return np.mean(aucs)


def neuralnet_mean_auc_score(X, y):
    kfold = KFold(shuffle=True)
    aucs = []
    for train, test in kfold.split(X):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        mlp = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            early_stopping=True,
        )
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict_proba(X_test)
        aucs.append(auc_score(y_test, y_pred))
    return np.mean(aucs), np.std(aucs)


def dt_mean_auc_score(y_pred, y_true):
    X = y_pred.reshape((-1, 1))
    dt = DecisionTreeClassifier(class_weight="balanced")
    dt.fit(X, y_true)
    y_pred = dt.predict_proba(X)
    return auc_score(y_true, y_pred)
