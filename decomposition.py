import os
import pickle

import numpy as np
from scipy.stats import kurtosis
from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import RandomForestClassifier
from sklearn.random_projection import SparseRandomProjection

from datasets import load_dataset
from evaluate import lr_mean_auc_score

N_COMPONENTS = [2, 5, 8, 10, 15, 32, 40, 50, 55]


class RFCA:
    def __init__(self):
        self.rf = RandomForestClassifier(n_jobs=16, random_state=0)

    def fit(self, X, y):
        self.rf.fit(X, y)
        self.ranked_features_ = (-self.rf.feature_importances_).argsort()
        self.feature_importances_ = self.rf.feature_importances_[self.ranked_features_]
        self.feature_importances_ /= self.feature_importances_.sum()

    def transform(self, X, n_components):
        return X[:, self.ranked_features_[:n_components]]

    def inverse_transform(self, X):
        n = X.shape[1]
        out = np.zeros((X.shape[0], len(self.ranked_features_)))
        out[:, self.ranked_features_[:n]] = X
        return out


def already_done(n, out):
    for item in out:
        if item[0] == n:
            return True
    return False


def train_pca(dataset):
    path = f"readings/pca_{dataset}.pkl"
    out = []
    if os.path.exists(path):
        out = pickle.load(open(path, "rb"))

    X, y = load_dataset(dataset)
    for n_components in N_COMPONENTS + [X.shape[1]]:
        if already_done(n_components, out):
            continue
        print(n_components)
        pca = PCA(n_components, random_state=0)
        X_trans = pca.fit_transform(X)
        X_recon = pca.inverse_transform(X_trans)
        error = ((X_recon - X) ** 2).sum(axis=1).mean()
        out.append(
            (
                n_components,
                error,
                np.sum(pca.explained_variance_ratio_),
                lr_mean_auc_score(X_trans, y),
            )
        )
    pickle.dump(out, open(path, "wb"))
    return out


def train_ica(dataset):
    path = f"readings/ica_{dataset}.pkl"
    out = []
    if os.path.exists(path):
        out = pickle.load(open(path, "rb"))

    X, y = load_dataset(dataset)
    for n_components in N_COMPONENTS + [X.shape[1]]:
        if already_done(n_components, out):
            continue
        print(n_components)
        ica = FastICA(n_components, random_state=0)
        X_trans = ica.fit_transform(X)
        X_recon = ica.inverse_transform(X_trans)
        error = ((X_recon - X) ** 2).sum(axis=1).mean()
        out.append(
            (
                n_components,
                error,
                np.abs(kurtosis(X_trans)).mean(),
                lr_mean_auc_score(X_trans, y),
            )
        )
    pickle.dump(out, open(path, "wb"))
    return out


def train_rca(dataset):
    path = f"readings/rca_{dataset}.pkl"
    out = []
    if os.path.exists(path):
        out = pickle.load(open(path, "rb"))

    X, y = load_dataset(dataset)
    for n_components in N_COMPONENTS + [X.shape[1]]:
        if already_done(n_components, out):
            continue
        print(n_components)
        errors = []
        aucs = []
        for i in range(5):
            rca = SparseRandomProjection(
                n_components,
                random_state=n_components * 5 + i,
            )
            X_trans = rca.fit_transform(X)
            X_recon = rca.inverse_transform(X_trans)
            errors.append(((X_recon - X) ** 2).sum(axis=1).mean())
            aucs.append(lr_mean_auc_score(X_trans, y))
        out.append(
            (
                n_components,
                np.mean(errors),
                np.std(errors),
                np.mean(aucs),
                np.std(aucs),
            )
        )
    pickle.dump(out, open(path, "wb"))
    return out


def train_rfca(dataset):
    path = f"readings/rfca_{dataset}.pkl"
    out = []
    if os.path.exists(path):
        out = pickle.load(open(path, "rb"))

    X, y = load_dataset(dataset)
    rfca = RFCA()
    rfca.fit(X, y)
    for n_components in N_COMPONENTS + [X.shape[1]]:
        if already_done(n_components, out):
            continue
        print(n_components)
        X_trans = rfca.transform(X, n_components)
        X_recon = rfca.inverse_transform(X_trans)
        error = ((X_recon - X) ** 2).sum(axis=1).mean()
        out.append(
            (
                n_components,
                error,
                rfca.feature_importances_[:n_components].sum(),
                lr_mean_auc_score(X_trans, y),
            )
        )
    pickle.dump(out, open(path, "wb"))
    return out


if __name__ == "__main__":
    for dataset in ["credit_score", "term_deposits"]:
        print(dataset)
        train_pca(dataset)
        train_ica(dataset)
        train_rca(dataset)
        train_rfca(dataset)
