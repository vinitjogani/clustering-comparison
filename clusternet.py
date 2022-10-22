import os
import pickle
from time import time
import numpy as np
from combined import *
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from evaluate import pr_auc_score
from sklearn.metrics import log_loss
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

KWARGS = dict(
    activation="relu",
    batch_size=128,
)


def none(X):
    return X


def model_generator(algo, n, replacement, soft):
    def fn(X):
        model = algo(n)
        model.fit(X)

        if soft:
            if hasattr(model, "transform"):
                preds = model.transform(X)
            else:
                preds = model.predict_proba(X)
        else:
            preds = model.predict(X)
            preds = np.eye(n)[preds]

        if replacement:
            return preds
        else:
            return np.hstack([X, preds])

    return fn


def get_models():
    for algo_name, algo, n in [("kmeans", KMeans, 30), ("gmm", GaussianMixture, 30)]:
        for replacement in [True, False]:
            for soft in [True, False]:
                r_key = "replace" if replacement else "add"
                s_key = "soft" if soft else "hard"
                key = f"{algo_name}_{r_key}_{s_key}"
                yield model_generator(algo, n, replacement, soft), key


def by_iterations(dataset):
    X, y = load_dataset(dataset)

    for algo, key in get_models():
        cache = f"readings/iter_{key}.pkl"

        if os.path.exists(cache):
            print("Skipping", cache)
            continue

        X_trans = algo(X)
        X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.2)
        model = MLPClassifier((512, 256), max_iter=2, warm_start=True, **KWARGS)

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
            model.set_params(max_iter=i * 3 + 2)

        out = {"train_loss": train_loss, "test_loss": test_loss}
        pickle.dump(out, open(cache, "wb"))


def by_training_size(dataset):
    X, y = load_dataset(dataset)

    for algo, key in get_models():
        cache = f"readings/trainsize_{key}.pkl"
        print(cache)
        if os.path.exists(cache):
            continue

        X_trans = algo(X)

        train_auc = []
        test_auc = []
        timings = []
        for size in [0.2, 0.4, 0.6, 0.8, 0.9]:
            X_train, X_test, y_train, y_test = train_test_split(
                X_trans, y, test_size=1 - size
            )

            model = MLPClassifier((512, 256), early_stopping=True, **KWARGS)

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
