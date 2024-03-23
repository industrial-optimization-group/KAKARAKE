import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def GaussianMixtureClusteringWithBIC(data: pd.DataFrame):
    data = StandardScaler().fit_transform(data)
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, min(11, len(data)))
    cv_types = ["spherical", "tied", "diag", "full"]
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = GaussianMixture(n_components=n_components, covariance_type=cv_type)
            gmm.fit(data)
            bic.append(gmm.score(data))
            # bic.append(gmm.bic(data))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    return best_gmm.predict(data)


def GaussianMixtureClusteringWithSilhouette(data: pd.DataFrame):
    X = StandardScaler().fit_transform(data)
    best_score = -np.infty
    best_labels = []
    n_components_range = range(1, min(11, len(data)))
    cv_types = ["spherical", "tied", "diag", "full"]
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = GaussianMixture(n_components=n_components, covariance_type=cv_type)
            labels = gmm.fit_predict(X)
            try:
                score = silhouette_score(X, labels, metric="cosine")
            except ValueError:
                score = -np.infty
            if score > best_score:
                best_score = score
                best_labels = labels
    # print(best_score)
    return best_labels


def DBSCANClustering(data: pd.DataFrame):
    X = StandardScaler().fit_transform(data)
    eps_options = np.linspace(0.01, 1, 20)
    best_score = -np.infty
    best_labels = [1] * len(X)
    for eps_option in eps_options:
        db = DBSCAN(eps=eps_option, min_samples=10, metric="cosine").fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        try:
            score = silhouette_score(X, labels, metric="cosine")
        except ValueError:
            score = -np.infty
        if score > best_score:
            best_score = score
            best_labels = labels
    # print((best_score, chosen_eps))
    return best_labels


def cluster(data: pd.DataFrame, algorithm: str = "DBSCAN", score: str = "silhoutte"):
    if not (score == "silhoutte" or score == "BIC"):
        raise ValueError()
    if not (algorithm == "GMM" or algorithm == "DBSCAN"):
        raise ValueError()
    if algorithm == "DBSCAN":
        return DBSCANClustering(data)
    elif score == "silhoutte":
        return GaussianMixtureClusteringWithSilhouette(data)
    else:
        return GaussianMixtureClusteringWithBIC(data)
