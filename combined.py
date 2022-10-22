import os
import pickle

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import calinski_harabasz_score, homogeneity_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.random_projection import SparseRandomProjection

from datasets import load_dataset
from decomposition import RFCA
from evaluate import dt_mean_auc_score


def ica(dataset, X, y, n=None):
    _c = {
        "credit_score": 30,
        "term_deposits": 50,
    }
    if n is None:
        n = _c[dataset]
    ica = FastICA(n, random_state=0)
    X_trans = ica.fit_transform(X)
    return X_trans


def pca(dataset, X, y):
    _c = {
        "credit_score": 30,
        "term_deposits": 30,
    }

    pca = PCA(_c[dataset], random_state=0)
    X_trans = pca.fit_transform(X)
    return X_trans


def rca(dataset, X, y):
    _c = {
        "credit_score": 40,
        "term_deposits": 40,
    }

    rca = SparseRandomProjection(_c[dataset])
    X_trans = rca.fit_transform(X)
    return X_trans


def rfca(dataset, X, y):
    _c = {
        "credit_score": 30,
        "term_deposits": 40,
    }

    rfca = RFCA()
    rfca.fit(X, y)
    X_trans = rfca.transform(X, _c[dataset])
    return X_trans


def clustering(X, y, algo, kwargs):
    n_clusters = [2, 4, 8, 16, 32, 48, 64]
    out = []
    for k in n_clusters:
        model = algo(k, **kwargs)
        y_pred = model.fit_predict(X)

        score = calinski_harabasz_score(X, y_pred)
        auc = dt_mean_auc_score(y_pred, y)
        silhouette = silhouette_score(X, y_pred, sample_size=10_000)
        homogeneity = homogeneity_score(y, y_pred)

        out.append((k, score, auc, silhouette, homogeneity))
        print(algo.__name__, k, out[-1])
    return out


def run_all():
    cluster_algos = {
        "kmeans": (KMeans, {}),
        "gmm": (GaussianMixture, {}),
    }

    for dataset in ["credit_score", "term_deposits"]:
        X, y = load_dataset(dataset)
        for decomp_algo in [pca, ica, rca, rfca]:
            X_trans = decomp_algo(dataset, X, y)
            for cluster_algo, (alg, kwargs) in cluster_algos.items():
                print(cluster_algo, decomp_algo.__name__, dataset)
                path = f"readings/{cluster_algo}_{decomp_algo.__name__}_{dataset}.pkl"
                if os.path.exists(path):
                    continue
                out = clustering(X_trans, y, alg, kwargs)
                pickle.dump(out, open(path, "wb"))


if __name__ == "__main__":
    run_all()
