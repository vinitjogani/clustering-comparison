import os
import pickle
from time import time

from combined import *
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from evaluate import pr_auc_score
from sklearn.metrics import log_loss


KWARGS = dict(
    activation="relu",
    batch_size=128,
)


def none(_, X, y):
    return X


def best_size(x):
    return {
        "pca": (512, 256),
        "rca": (512, 256),
        "ica": (512, 256, 128),
        "rfca": (256, 128),
    }.get(x, (512, 256, 128))


def by_iterations(dataset):
    X, y = load_dataset(dataset)

    for algo in [pca, ica, rca, rfca, none]:
        X_trans = algo(dataset, X, y)

        X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.2)

        model = MLPClassifier(
            best_size(algo.__name__), max_iter=100, warm_start=True, **KWARGS
        )
        cache = f"readings/iter_{algo.__name__}_{dataset}.pkl"

        if os.path.exists(cache):
            continue

        train_loss = []
        test_loss = []

        for i in range(5):
            print(i)
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_train)
            loss = log_loss(y_train, y_pred)
            train_loss.append(loss)
            y_pred = model.predict_proba(X_test)
            loss = log_loss(y_test, y_pred)
            test_loss.append(loss)
            model.set_params(max_iter=i * 100 + 100)

        out = {"train_loss": train_loss, "test_loss": test_loss}
        pickle.dump(out, open(cache, "wb"))


def by_training_size(dataset):
    X, y = load_dataset(dataset)

    for algo in [pca, ica, rca, rfca, none]:
        X_trans = algo(dataset, X, y)

        cache = f"readings/trainsize_{algo.__name__}_{dataset}.pkl"
        if os.path.exists(cache):
            continue

        train_auc = []
        test_auc = []
        timings = []
        for size in [0.2, 0.4, 0.6, 0.8, 1.0]:
            X_train, X_test, y_train, y_test = train_test_split(
                X_trans, y, test_size=1 - size
            )

            model = MLPClassifier(
                best_size(algo.__name__), early_stopping=True, **KWARGS
            )

            start = time()
            model.fit(X_train, y_train)
            elapsed = time() - start
            timings.append(elapsed)

            y_pred = model.predict_proba(X_train)
            auc = pr_auc_score(y_train, y_pred)
            train_auc.append(auc)
            y_pred = model.predict_proba(X_test)
            auc = pr_auc_score(y_test, y_pred)
            test_auc.append(auc)

        out = {"train_aucs": train_auc, "test_aucs": test_auc, "timings": timings}
        pickle.dump(out, open(cache, "wb"))


if __name__ == "__main__":
    by_iterations("credit_score")
    by_training_size("credit_score")
