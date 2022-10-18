import os
import pickle

from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from sklearn.mixture import GaussianMixture

from datasets import *


def train_algo(dataset, algo):
    X, _ = load_dataset(dataset)
    n_clusters = [2, 4, 8, 16, 32, 48, 64]
    out = {}
    for k in n_clusters:
        model = algo(k)
        y_pred = model.fit_predict(X)
        score = calinski_harabasz_score(X, y_pred)
        out[k] = (k, model, score)
        print(out[k])
    return out


def train_kmeans(dataset):
    path = f"readings/kmeans_{dataset}.pkl"
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    out = train_algo(dataset, KMeans)
    pickle.dump(out, open(path, "wb"))
    return out


def train_gmm(dataset):
    path = f"readings/gmm_{dataset}.pkl"
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    out = train_algo(dataset, GaussianMixture)
    pickle.dump(out, open(path, "wb"))
    return out


if __name__ == "__main__":
    for dataset in ["credit_score", "term_deposits"]:
        print(dataset)
        train_kmeans(dataset)
        train_gmm(dataset)
