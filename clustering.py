"""Clusterer wrappers that share a common fit_predict + centroids interface.

Trainers depend on this abstraction so a new clustering method only requires
adding a new class here, not duplicating the whole training script.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


class Clusterer(Protocol):
    name: str

    def fit_predict(self, X: np.ndarray) -> np.ndarray: ...

    @property
    def centroids(self) -> np.ndarray: ...


@dataclass
class GMMClusterer:
    n_components: int
    random_state: int = 42
    n_init: int = 5
    name: str = "GMM"

    def __post_init__(self) -> None:
        self._gmm = GaussianMixture(
            n_components=self.n_components,
            random_state=self.random_state,
            n_init=self.n_init,
        )

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self._gmm.fit_predict(X)

    @property
    def centroids(self) -> np.ndarray:
        return self._gmm.means_


@dataclass
class KMeansClusterer:
    n_clusters: int
    random_state: int = 42
    n_init: int = 10
    name: str = "K-Means"

    def __post_init__(self) -> None:
        self._kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
        )

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self._kmeans.fit_predict(X)

    @property
    def centroids(self) -> np.ndarray:
        return self._kmeans.cluster_centers_


def build_clusterer(method: str, n_states: int, **kwargs) -> Clusterer:
    """Factory: ``method`` is 'gmm' or 'kmeans'."""
    method = method.lower()
    if method == "gmm":
        return GMMClusterer(n_components=n_states, **kwargs)
    if method == "kmeans":
        return KMeansClusterer(n_clusters=n_states, **kwargs)
    raise ValueError(f"Unknown clustering method: {method!r}")
