from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import SparseRandomProjection
from decomposition import RFCA


def ica(dataset, X):
    _c = {
        "credit_score": 5,
        "term_deposits": 50,
    }

    ica = FastICA(_c[dataset], random_state=0)
    X_trans = ica.fit_transform(X)
    return X_trans
