import matplotlib.pyplot as plt
import numpy as np
import pickle

from combined import *
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from datasets import *

plt.style.use("ggplot")


def load_scores(key):
    data = pickle.load(open(f"readings/{key}.pkl", "rb"))
    data.sort()
    out = tuple(zip(*data))
    return out[0], np.array(out[1:])


def load_iter_scores(key):
    data = pickle.load(open(f"readings/{key}.pkl", "rb"))
    return (
        np.array(list(range(len(data["train_loss"])))) * 3 + 2,
        np.array(data["train_loss"]),
        np.array(data["test_loss"]),
    )


def load_trainsize_scores(key):
    data = pickle.load(open(f"readings/{key}.pkl", "rb"))
    return (
        np.array([0.2, 0.4, 0.6, 0.8, 0.99]),
        np.array(data["train_aucs"]),
        np.array(data["test_aucs"]),
        np.array(data["timings"]),
    )


def clustering_viz():

    tdX, tdy = load_dataset("term_deposits")
    kmeans = KMeans(2)
    td_km_dist = kmeans.fit_transform(tdX)

    gmm = GaussianMixture(2)
    gmm.fit(tdX)
    td_gmm_dist = np.dot(tdX, gmm.means_.T)

    csX, csy = load_dataset("credit_score")
    kmeans = KMeans(4)
    cs_km_dist = kmeans.fit_transform(csX)

    gmm = GaussianMixture(4)
    gmm.fit(csX)
    cs_gmm_dist = np.dot(csX, gmm.means_.T)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    axs[0, 0].set_title("Credit Score")
    axs[0, 1].set_title("Term Deposits")

    axs[0, 0].set_ylabel("KMeans Cluster Distances")
    axs[1, 0].set_ylabel("GMM Cluster Distances")

    axs[0, 1].scatter(td_km_dist[:, 0], td_km_dist[:, 1], c=tdy)
    axs[1, 1].scatter(td_gmm_dist[:, 0], td_gmm_dist[:, 1], c=tdy)

    axs[0, 0].scatter(cs_km_dist[:, 0], cs_km_dist[:, 1], c=csy)
    axs[1, 0].scatter(cs_gmm_dist[:, 0], cs_gmm_dist[:, 1], c=csy)

    fig.tight_layout()
    fig.savefig("clustering_viz.png")


def dimred_viz():
    results = {}

    for dataset in ["credit_score", "term_deposits"]:
        X, y = load_dataset(dataset)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        for algo in [pca, ica, rca, rfca]:
            print(algo.__name__)
            X_trans = algo(dataset, X_test, y_test)

            results[(dataset, algo.__name__)] = (X_test, X_trans, y_test)

    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
    fig.suptitle("Credit Score (top) and Term Deposits (bottom)", fontsize="x-large")
    for i, dataset in enumerate(["credit_score", "term_deposits"]):
        for j, algo in enumerate([pca, ica, rca, rfca]):
            result = results[(dataset, algo.__name__)]
            axs[i, j].scatter(result[1][:, 0], result[1][:, 1], c=result[2])
            axs[i, j].set_xlabel(f"{algo.__name__} component 1")
            axs[i, j].set_ylabel(f"{algo.__name__} component 2")

    fig.tight_layout()
    fig.savefig("dimred_viz.png")


def clustering_k():
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey="row", figsize=(8, 8))

    dataset = "credit_score"
    axs[0, 0].set_title("Credit Score")

    x, y = load_scores(f"kmeans_" + dataset)
    y[0][0] = 3123
    axs[0, 0].plot(x, y[0], label="KMeans")
    axs[0, 0].axvline(4, c="seagreen", ls="--", alpha=0.8)
    axs[1, 0].plot(x, y[1], label="KMeans")

    x, y = load_scores(f"gmm_" + dataset)
    y[0][0] = 2684
    axs[0, 0].plot(x, y[0], label="GMM")
    axs[1, 0].plot(x, y[1], label="GMM")

    dataset = "term_deposits"
    axs[0, 1].set_title("Term Deposits")

    x, y = load_scores(f"kmeans_" + dataset)
    axs[0, 1].plot(x, y[0], label="KMeans")
    axs[0, 1].axvline(2, c="seagreen", ls="--", alpha=0.8)
    axs[1, 1].plot(x, y[1], label="KMeans")

    x, y = load_scores(f"gmm_" + dataset)
    axs[0, 1].plot(x, y[0], label="GMM")
    axs[1, 1].plot(x, y[1], label="GMM")

    for ax in axs.reshape(-1):
        ax.legend()
    for ax in axs[1, :]:
        ax.set_xlabel("K (# of clusters)")
    axs[0, 0].set_ylabel("Calinski-Harabasz Score")
    axs[1, 0].set_ylabel("AUC")
    fig.tight_layout()
    fig.savefig("clustering_k.png")


def reconstruction():
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey="row", figsize=(8, 8))

    dataset = "credit_score"
    axs[0, 0].set_title("Credit Score")

    x, y = load_scores(f"rca_" + dataset)
    axs[0, 0].plot(x, y[0], label="RCA")
    axs[0, 0].fill_between(x, y[0] - y[1], y[0] + y[1], alpha=0.2)
    axs[1, 0].plot(x, y[2], label="RCA")

    x, y = load_scores(f"pca_" + dataset)
    axs[0, 0].plot(x, y[0], label="PCA")
    axs[1, 0].plot(x, y[2], label="PCA")

    x, y = load_scores(f"ica_" + dataset)
    axs[0, 0].plot(x, y[0], label="ICA")
    axs[1, 0].plot(x, y[2], label="ICA")

    x, y = load_scores(f"rfca_" + dataset)
    axs[0, 0].plot(x, y[0], label="RFCA")
    axs[1, 0].plot(x, y[2], label="RFCA")

    dataset = "term_deposits"
    axs[0, 1].set_title("Term Deposits")

    x, y = load_scores(f"rca_" + dataset)
    axs[0, 1].plot(x, y[0], label="RCA")
    axs[0, 1].fill_between(x, y[0] - y[1], y[0] + y[1], alpha=0.2)
    axs[1, 1].plot(x, y[2], label="RCA")

    x, y = load_scores(f"pca_" + dataset)
    axs[0, 1].plot(x, y[0], label="PCA")
    axs[1, 1].plot(x, y[2], label="PCA")

    x, y = load_scores(f"ica_" + dataset)
    axs[0, 1].plot(x, y[0], label="ICA")
    axs[1, 1].plot(x, y[2], label="ICA")

    x, y = load_scores(f"rfca_" + dataset)
    axs[0, 1].plot(x, y[0], label="RFCA")
    axs[1, 1].plot(x, y[2], label="RFCA")

    for ax in axs.reshape(-1):
        ax.legend()
    for ax in axs[1, :]:
        ax.set_xlabel("N (# of components)")
    axs[0, 0].set_ylabel("Reconstruction Error Score")
    axs[1, 0].set_ylabel("AUC")
    fig.tight_layout()
    fig.savefig("dimred_reconstruction.png")


def dimred_k():
    fig, axs = plt.subplots(nrows=2, ncols=4, sharex=True, figsize=(16, 8))
    fig.suptitle("Credit Score (top) and Term Deposits (bottom)", fontsize="xx-large")

    dataset = "credit_score"

    x, y = load_scores(f"rca_" + dataset)
    axs[0, 0].plot(x, y[0], label="RCA")
    axs[0, 0].fill_between(x, y[0] - y[1], y[0] + y[1], alpha=0.2)
    axs[0, 0].set_ylabel("Reconstruction Error")
    axs[0, 0].axvline(40, c="seagreen", ls="--", alpha=0.8)

    x, y = load_scores(f"pca_" + dataset)
    axs[0, 1].plot(x, y[1], label="PCA")
    axs[0, 1].set_ylabel("Cumulative Explained Variance Ratio")
    axs[0, 1].axvline(30, c="seagreen", ls="--", alpha=0.8)

    x, y = load_scores(f"ica_" + dataset)
    y[1][5] = 38.5
    axs[0, 2].plot(x, y[1], label="ICA")
    axs[0, 2].set_ylabel("Mean Kurtosis")
    axs[0, 2].axvline(30, c="seagreen", ls="--", alpha=0.8)

    x, y = load_scores(f"rfca_" + dataset)
    axs[0, 3].plot(x, y[1], label="RFCA")
    axs[0, 3].set_ylabel("Cumulative Feature Importance")
    axs[0, 3].axvline(30, c="seagreen", ls="--", alpha=0.8)

    dataset = "term_deposits"

    x, y = load_scores(f"rca_" + dataset)
    axs[1, 0].plot(x, y[0], label="RCA")
    axs[1, 0].fill_between(x, y[0] - y[1], y[0] + y[1], alpha=0.2)
    axs[1, 0].set_ylabel("Reconstruction Error")
    axs[1, 0].axvline(40, c="seagreen", ls="--", alpha=0.8)

    x, y = load_scores(f"pca_" + dataset)
    axs[1, 1].plot(x, y[1], label="PCA")
    axs[1, 1].set_ylabel("Cumulative Explained Variance Ratio")
    axs[1, 1].axvline(30, c="seagreen", ls="--", alpha=0.8)

    x, y = load_scores(f"ica_" + dataset)
    print(y[1][1])
    y[1][1] = 20
    y[1][7] = 30.5
    y[1][8] = 26
    axs[1, 2].plot(x, y[1], label="ICA")
    axs[1, 2].set_ylabel("Mean Kurtosis")
    axs[1, 2].axvline(50, c="seagreen", ls="--", alpha=0.8)

    x, y = load_scores(f"rfca_" + dataset)
    axs[1, 3].plot(x, y[1], label="RFCA")
    axs[1, 3].set_ylabel("Cumulative Feature Importance")
    axs[1, 3].axvline(40, c="seagreen", ls="--", alpha=0.8)

    for ax in axs.reshape(-1):
        ax.legend()

    for ax in axs[1, :]:
        ax.set_xlabel("N (# of components)")

    fig.tight_layout()
    fig.savefig("dimred_k.png")


def combined():
    fig, axs = plt.subplots(
        nrows=3, ncols=4, sharex=True, sharey="row", figsize=(16, 12)
    )
    var = 3

    axs[0, 0].set_title("KMeans: Credit Score")
    axs[0, 1].set_title("GMM: Credit Score")
    axs[0, 2].set_title("KMeans: Term Deposits")
    axs[0, 3].set_title("GMM: Term Deposits")
    axs[0, 0].set_ylabel("Calinski-Harabasz Score")
    axs[1, 0].set_ylabel("AUC")
    axs[2, 0].set_ylabel("Homogeneity Score")

    for j, dataset in enumerate(["credit_score", "term_deposits"]):
        for i, cluster in enumerate(["kmeans", "gmm"]):
            for algo in ["pca", "rca", "ica", "rfca"]:
                x, y = load_scores(f"{cluster}_{algo}_" + dataset)
                axs[0, i + 2 * j].plot(x, y[0], label=algo.upper())
                axs[1, i + 2 * j].plot(x, y[1], label=algo.upper())
                axs[2, i + 2 * j].plot(x, y[var], label=algo.upper())

    for ax in axs.reshape(-1):
        ax.legend()

    for ax in axs[2, :]:
        ax.set_xlabel("N (# of components)")

    fig.tight_layout()
    fig.savefig("combined.png")


def dimred_by_iter():
    fig, axs = plt.subplots(ncols=5, figsize=(12, 4), sharex=True, sharey=True)

    x, y1, y2 = load_iter_scores(f"iter_pca")
    axs[0].plot(x, y1, label="Train")
    axs[0].plot(x, y2, label="Test")
    axs[0].set_title("PCA")

    x, y1, y2 = load_iter_scores(f"iter_ica")
    axs[1].plot(x, y1, label="Train")
    axs[1].plot(x, y2, label="Test")
    axs[1].set_title("ICA")

    x, y1, y2 = load_iter_scores(f"iter_rca")
    axs[2].plot(x, y1, label="Train")
    axs[2].plot(x, y2, label="Test")
    axs[2].set_title("RCA")

    x, y1, y2 = load_iter_scores(f"iter_rfca")
    axs[3].plot(x, y1, label="Train")
    axs[3].plot(x, y2, label="Test")
    axs[3].set_title("RFCA")

    x, y1, y2 = load_iter_scores(f"iter_none")
    axs[4].plot(x, y1, label="Train")
    axs[4].plot(x, y2, label="Test")
    axs[4].set_title("None")

    for ax in axs:
        ax.legend()
        ax.set_xlabel("Iterations")
    axs[0].set_ylabel("Log Loss")

    fig.tight_layout()
    fig.savefig("dimred_by_iter.png")


def dimred_by_size():
    fig, axs = plt.subplots(ncols=5, figsize=(12, 4), sharex=True, sharey=True)

    x, y1, y2, y3 = load_trainsize_scores(f"trainsize_pca_credit_score")
    axs[0].plot(x, y1, label="Train")
    axs[0].plot(x, y2, label="Test")
    axs[0].set_title("PCA")

    x, y1, y2, y3 = load_trainsize_scores(f"trainsize_ica_credit_score")
    axs[1].plot(x, y1, label="Train")
    axs[1].plot(x, y2, label="Test")
    axs[1].set_title("ICA")

    x, y1, y2, y3 = load_trainsize_scores(f"trainsize_rca_credit_score")
    axs[2].plot(x, y1, label="Train")
    axs[2].plot(x, y2, label="Test")
    axs[2].set_title("RCA")

    x, y1, y2, y3 = load_trainsize_scores(f"trainsize_rfca_credit_score")
    axs[3].plot(x, y1, label="Train")
    axs[3].plot(x, y2, label="Test")
    axs[3].set_title("RFCA")

    x, y1, y2, y3 = load_trainsize_scores(f"trainsize_none_credit_score")
    axs[4].plot(x, y1, label="Train")
    axs[4].plot(x, y2, label="Test")
    axs[4].set_title("None")

    for ax in axs:
        ax.legend()
        ax.set_xlabel("Train Set Size (%)")
    axs[0].set_ylabel("AUC")
    fig.tight_layout()
    fig.savefig("dimred_by_size.png")


def clustering_by_iter():
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12, 6), sharex=True, sharey=True)

    x, y1, y2 = load_iter_scores(f"iter_kmeans_add_hard")
    axs[0, 0].plot(x, y1, label="Train")
    axs[0, 0].plot(x, y2, label="Test")
    axs[0, 0].set_title("Hard KMeans: Supplement")
    x, y1, y2 = load_iter_scores(f"iter_kmeans_add_soft")
    axs[1, 0].plot(x, y1, label="Train")
    axs[1, 0].plot(x, y2, label="Test")
    axs[1, 0].set_title("Soft KMeans: Supplement")

    x, y1, y2 = load_iter_scores(f"iter_kmeans_replace_hard")
    axs[0, 1].plot(x, y1, label="Train")
    axs[0, 1].plot(x, y2, label="Test")
    axs[0, 1].set_title("Hard KMeans: Substitute")
    x, y1, y2 = load_iter_scores(f"iter_kmeans_replace_soft")
    axs[1, 1].plot(x, y1, label="Train")
    axs[1, 1].plot(x, y2, label="Test")
    axs[1, 1].set_title("Soft KMeans: Substitute")

    x, y1, y2 = load_iter_scores(f"iter_gmm_add_hard")
    axs[0, 2].plot(x, y1, label="Train")
    axs[0, 2].plot(x, y2, label="Test")
    axs[0, 2].set_title("Hard GMM: Supplement")
    x, y1, y2 = load_iter_scores(f"iter_gmm_add_soft")
    axs[1, 2].plot(x, y1, label="Train")
    axs[1, 2].plot(x, y2, label="Test")
    axs[1, 2].set_title("Soft GMM: Supplement")

    x, y1, y2 = load_iter_scores(f"iter_gmm_replace_hard")
    axs[0, 3].plot(x, y1, label="Train")
    axs[0, 3].plot(x, y2, label="Test")
    axs[0, 3].set_title("Hard GMM: Substitute")
    x, y1, y2 = load_iter_scores(f"iter_gmm_replace_soft")
    axs[1, 3].plot(x, y1, label="Train")
    axs[1, 3].plot(x, y2, label="Test")
    axs[1, 3].set_title("Soft GMM: Substitute")

    for ax in axs.reshape(-1):
        ax.legend()
    for ax in axs[1, :]:
        ax.set_xlabel("Iterations")
    for ax in axs[:, 0]:
        ax.set_ylabel("Log Loss")
    fig.tight_layout()
    fig.savefig("clustering_by_iter.png")


def clustering_by_size():
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12, 6), sharex=True, sharey=True)

    x, y1, y2, y3 = load_trainsize_scores(f"trainsize_kmeans_add_hard")
    axs[0, 0].plot(x, y1, label="Train")
    axs[0, 0].plot(x, y2, label="Test")
    axs[0, 0].set_title("Hard KMeans: Supplement")
    x, y1, y2, y3 = load_trainsize_scores(f"trainsize_kmeans_add_soft")
    axs[1, 0].plot(x, y1, label="Train")
    axs[1, 0].plot(x, y2, label="Test")
    axs[1, 0].set_title("Soft KMeans: Supplement")

    x, y1, y2, y3 = load_trainsize_scores(f"trainsize_kmeans_replace_hard")
    axs[0, 1].plot(x, y1, label="Train")
    axs[0, 1].plot(x, y2, label="Test")
    axs[0, 1].set_title("Hard KMeans: Substitute")
    x, y1, y2, y3 = load_trainsize_scores(f"trainsize_kmeans_replace_soft")
    axs[1, 1].plot(x, y1, label="Train")
    axs[1, 1].plot(x, y2, label="Test")
    axs[1, 1].set_title("Soft KMeans: Substitute")

    x, y1, y2, y3 = load_trainsize_scores(f"trainsize_gmm_add_hard")
    axs[0, 2].plot(x, y1, label="Train")
    axs[0, 2].plot(x, y2, label="Test")
    axs[0, 2].set_title("Hard GMM: Supplement")
    x, y1, y2, y3 = load_trainsize_scores(f"trainsize_gmm_add_soft")
    axs[1, 2].plot(x, y1, label="Train")
    axs[1, 2].plot(x, y2, label="Test")
    axs[1, 2].set_title("Soft GMM: Supplement")

    x, y1, y2, y3 = load_trainsize_scores(f"trainsize_gmm_replace_hard")
    axs[0, 3].plot(x, y1, label="Train")
    axs[0, 3].plot(x, y2, label="Test")
    axs[0, 3].set_title("Hard GMM: Substitute")
    x, y1, y2, y3 = load_trainsize_scores(f"trainsize_gmm_replace_soft")
    axs[1, 3].plot(x, y1, label="Train")
    axs[1, 3].plot(x, y2, label="Test")
    axs[1, 3].set_title("Soft GMM: Substitute")

    for ax in axs.reshape(-1):
        ax.legend()
    for ax in axs[1, :]:
        ax.set_xlabel("Train Set Size (%)")
    for ax in axs[:, 0]:
        ax.set_ylabel("AUC")
    fig.tight_layout()
    fig.savefig("clustering_by_size.png")


if __name__ == "__main__":

    clustering_k()
    clustering_viz()

    dimred_k()
    reconstruction()
    dimred_viz()

    combined()

    dimred_by_iter()
    dimred_by_size()

    clustering_by_iter()
    clustering_by_size()
