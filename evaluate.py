import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier


def pr_auc_score(y_true, y_score):
    aucs = []
    counts = []
    for class_ in range(y_score.shape[1]):
        precision, recall, _ = precision_recall_curve(
            y_true == class_,
            y_score[:, class_],
        )
        aucs.append(auc(recall, precision))
        counts.append((y_true == class_).sum())

    return sum(aucs) / len(aucs)


def lr_mean_auc_score(X, y):
    kfold = KFold(shuffle=True)
    aucs = []
    for train, test in kfold.split(X):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict_proba(X_test)
        aucs.append(pr_auc_score(y_test, y_pred))
    return np.mean(aucs)


def dt_mean_auc_score(y_pred, y_true):
    X = y_pred.reshape((-1, 1))
    dt = DecisionTreeClassifier(class_weight="balanced")
    dt.fit(X, y_true)
    y_pred = dt.predict_proba(X)
    return pr_auc_score(y_true, y_pred)
