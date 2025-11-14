import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from typing import Iterable, List, Optional, Literal, Union
from hmmlearn.hmm import GaussianHMM
import random


@dataclass
class WKMeansResult:
    centroids: np.ndarray  # shape (k, p)
    labels: Union[np.ndarray, pd.Series]  # shape (n,)
    losses: List[float]
    iter: int


class WassersteinKMeans:
    def __init__(self, n_clusters=2, p=2, max_iter=100, tol=1e-6, random_state=None):
        self.n_clusters = n_clusters
        self.p = p
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids_ = None
        self.labels_ = None
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

    def _wasserstein_empirical(self, alpha: np.ndarray, beta: np.ndarray, p=2):
        """
        Compute empirical Wasserstein-p distance between two empirical measures.

        Parameters
        ----------
        alpha : array-like
            Samples from distribution μ.
        beta : array-like
            Samples from distribution ν.
        p : int
            Order of the Wasserstein distance (default=2).

        Returns
        -------
        Wp : float
            Wasserstein-p distance.
        """
        alpha = np.sort(np.array(alpha))
        beta = np.sort(np.array(beta))

        N = min(len(alpha), len(beta))
        return (np.mean(np.abs(alpha[:N] - beta[:N]) ** p)) ** (1 / p)

    def _wasserstein_barycenter(self, samples: np.ndarray, p=2):
        """
        Compute the Wasserstein barycenter (centroid) of a list of 1D empirical samples.
        - For p=1: coordinate-wise median of order statistics.
        - For p=2: coordinate-wise mean of order statistics.
        """
        N = len(samples[0])
        # Stack sorted samples
        S = np.vstack([np.sort(s)[:N] for s in samples])
        if p == 1:
            return np.median(S, axis=0)
        else:
            return np.mean(S, axis=0)

    @staticmethod
    def _prepare_segments(segments: Iterable) -> tuple[List[np.ndarray], Optional[pd.Index]]:
        if isinstance(segments, pd.Series):
            index = segments.index
            iterable = segments.values
        else:
            index = None
            iterable = segments
        prepared = [np.asarray(seg, dtype=float).ravel() for seg in iterable]
        return prepared, index

    def fit(self, segments: List[np.ndarray], initial_centroids: Optional[List[np.ndarray]] = None) -> WKMeansResult:
        """
        Run WK-means on a list of empirical distributions (arrays of equal length).
        """
        segments, index = self._prepare_segments(segments)
        n_samples = len(segments)
        if n_samples < self.n_clusters:
            raise ValueError("Number of samples must be >= number of clusters.")

        # Allow for warm-start training
        if initial_centroids is not None:
            if len(initial_centroids) != self.n_clusters:
                raise ValueError("initial_centroids must have length equal to n_clusters")
            self.centroids_ = [np.array(c, copy=True) for c in initial_centroids]
        else:
            self.centroids_ = [segments[i] for i in np.random.choice(n_samples, self.n_clusters, replace=False)]
        self.labels_ = np.zeros(n_samples, dtype=int)

        losses = []
        iteration = 0
        while iteration < self.max_iter:
            # Assignment step
            for i, sample in enumerate(segments):
                distances = [self._wasserstein_empirical(sample, c, p=self.p) for c in self.centroids_]
                self.labels_[i] = np.argmin(distances)

            # Update step
            new_centroids = []
            for j in range(self.n_clusters):
                cluster_members = [segments[i] for i in range(n_samples) if self.labels_[i] == j]
                if cluster_members:
                    new_centroids.append(self._wasserstein_barycenter(cluster_members, p=self.p))
                else:
                    # Handle empty cluster by reinitializing
                    new_centroids.append(random.choice(segments))

            # Check convergence
            shift = sum(self._wasserstein_empirical(self.centroids_[j], new_centroids[j], p=self.p) for j in range(self.n_clusters))
            self.centroids_ = new_centroids

            iteration += 1
            losses.append(shift)

            if shift < self.tol:
                break
        if iteration == self.max_iter:
            print(f"Warning: WK-means algorithm may not converge after {self.max_iter} iterations")

        # Sort the labels by variance of centroid
        centroid_vars = np.var(self.centroids_, axis=1)
        sorted_indices = np.argsort(centroid_vars)
        self.centroids_ = [self.centroids_[i] for i in sorted_indices]
        self.labels_ = np.array([np.where(sorted_indices == lbl)[0][0] for lbl in self.labels_])
        if index is not None:
            self.labels_ = pd.Series(self.labels_, index=index)

        return WKMeansResult(centroids=np.array(self.centroids_), labels=self.labels_, losses=losses, iter=iteration)

    def predict(self, segments: List[np.ndarray]) -> np.ndarray:
        """
        Assign new samples to clusters.
        """
        if self.centroids_ is None:
            raise RuntimeError("Model not fitted yet.")
        segments, index = self._prepare_segments(segments)
        if len(segments[0]) != len(self.centroids_[0]):
            raise ValueError("Input samples must have the same length as centroids.")

        labels = []
        for sample in segments:
            distances = [self._wasserstein_empirical(sample, c, p=self.p) for c in self.centroids_]
            labels.append(np.argmin(distances))
        labels = np.array(labels)
        if index is not None:
            return pd.Series(labels, index=index)
        return labels

    def plot_centroids_cdf(self, title="WK-means Centroids CDFs"):
        """
        Plot the CDFs of WK-means centroids.

        Parameters
        ----------
        title : str
            Plot title.
        """
        if self.centroids_ is None:
            raise RuntimeError("Model not fitted yet.")

        plt.figure(figsize=(8, 6))
        palette = {0: "#4477AA", 1: "#228833", 2: "#EE6677", 3: "#CCBB44", 4: "#66CCEE", 5: "#AA3377", 6: "#BBBBBB"}
        cmap = plt.get_cmap("tab10", len(self.centroids_))
        for i, c in enumerate(self.centroids_):
            sorted_c = np.sort(c)
            cdf = np.arange(1, len(c) + 1) / len(c)
            color = palette.get(i, cmap(i))
            plt.plot(sorted_c, cdf, label=f"Centroid {i}", color=color)

        plt.title(title)
        plt.xlabel("Value")
        plt.ylabel("CDF")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()


@dataclass
class MKMeansResult:
    centroids: np.ndarray  # shape (k, p)
    labels: Union[np.ndarray, pd.Series]
    losses: List[float]  # L2 shift per iteration
    iters: int


class MomentKMeans:
    """
    K-means on the first `p_moments` raw moments of segments.
    - Uses Euclidean distance in moment space (after optional z-scoring).
    - k-means++ initialization.

    Fit input: a list of segments (1D arrays of equal length recommended).
    """

    def __init__(
        self,
        n_clusters: int = 2,
        p_moments: int = 4,
        standardize: bool = True,
        max_iter: int = 100,
        tol: float = 1e-6,
        init: Literal["kmeans++", "random"] = "kmeans++",
        random_state: Optional[int] = None,
    ):
        if n_clusters < 2:
            raise ValueError("n_clusters must be >= 2")
        if p_moments < 1:
            raise ValueError("p_moments must be >= 1")
        self.k = n_clusters
        self.p = p_moments
        self.standardize = standardize
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

        self.centroids_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self._features_: Optional[np.ndarray] = None  # cached feature matrix (n, p)

    @staticmethod
    def _prepare_segments(segments: Iterable) -> tuple[List[np.ndarray], Optional[pd.Index]]:
        if isinstance(segments, pd.Series):
            index = segments.index
            iterable = segments.values
        else:
            index = None
            iterable = segments
        prepared = [np.asarray(seg, dtype=float).ravel() for seg in iterable]
        return prepared, index

    def _moments_vector(self, x: np.ndarray, p: int) -> np.ndarray:
        """
        First p raw moments of a 1D sample x (finite-length empirical distribution).
        m_n = E[X^n] estimated by sample average.
        Returns shape (p,)
        """
        x = np.asarray(x, dtype=float).ravel()
        return np.array([np.mean(x**n) for n in range(1, p + 1)], dtype=float)

    def _zscore_columns(self, M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Column-wise z-score standardization (mean 0, var 1)."""
        mu = M.mean(axis=0)
        sd = M.std(axis=0)
        return (M - mu) / np.maximum(sd, eps)

    def _build_features(self, segments: List[np.ndarray]) -> np.ndarray:
        F = np.vstack([self._moments_vector(s, self.p) for s in segments]).astype(float)
        return self._zscore_columns(F) if self.standardize else F

    def _init_centroids(self, F: np.ndarray) -> np.ndarray:
        n = F.shape[0]
        if self.init == "random":
            idx = np.random.choice(n, self.k, replace=False)
            return F[idx].copy()

        # k-means++
        centroids = []
        idx0 = random.randrange(n)
        centroids.append(F[idx0])
        for _ in range(1, self.k):
            d2 = np.min(((F[:, None, :] - np.array(centroids)[None, :, :]) ** 2).sum(axis=2), axis=1)
            probs = d2 / d2.sum() if d2.sum() > 0 else np.ones(n) / n
            idx_next = np.random.choice(n, p=probs)
            centroids.append(F[idx_next])
        return np.vstack(centroids)

    def fit(self, segments: List[np.ndarray]) -> MKMeansResult:
        segments, index = self._prepare_segments(segments)
        if len(segments) < self.k:
            raise ValueError("Number of samples must be >= n_clusters.")
        F = self._build_features(segments)  # (n, p)
        self._features_ = F
        C = self._init_centroids(F)  # (k, p)
        labels = np.zeros(F.shape[0], dtype=int)
        losses: List[float] = []

        for it in range(1, self.max_iter + 1):
            # assign
            d2 = ((F[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)  # (n, k)
            labels = d2.argmin(axis=1)

            # update
            newC = np.zeros_like(C)
            for j in range(self.k):
                mask = labels == j
                if not np.any(mask):
                    # re-seed empty cluster
                    newC[j] = F[np.random.randint(0, F.shape[0])]
                else:
                    newC[j] = F[mask].mean(axis=0)

            # shift (L2 across centroids)
            shift = float(np.linalg.norm(C - newC))
            losses.append(shift)
            C = newC
            if shift < self.tol:
                break

        # Sort the labels by variance of centroid
        centroid_vars = np.var(C, axis=1)
        sorted_indices = np.argsort(centroid_vars)
        C = np.array([C[i] for i in sorted_indices])
        labels = np.array([np.where(sorted_indices == lbl)[0][0] for lbl in labels])

        self.centroids_ = C
        self.labels_ = labels
        label_output: Union[np.ndarray, pd.Series]
        if index is not None:
            label_output = pd.Series(labels, index=index)
        else:
            label_output = labels
        return MKMeansResult(C, label_output, losses, it)

    def predict(self, segments: List[np.ndarray]) -> Union[np.ndarray, pd.Series]:
        if self.centroids_ is None:
            raise RuntimeError("Model not fitted yet.")
        segments, index = self._prepare_segments(segments)
        if len(segments[0]) != len(self.centroids_[0]):
            raise ValueError("Input samples must have the same length as centroids.")

        F = self._build_features(segments)
        d2 = ((F[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)
        preds = d2.argmin(axis=1)
        if index is not None:
            return pd.Series(preds, index=index)
        return preds


@dataclass
class HMMClusteringResult:
    labels: Union[np.ndarray, pd.Series]
    means: List[float]
    stds: List[float]
    periods: List[int]


class HMMClustering:
    def __init__(self, n_states=3, covariance_type="full", random_state=None):
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.labels_: Optional[np.ndarray] = None
        self.model_: Optional[GaussianHMM] = None

    def _prepare_observations(self, X):
        if isinstance(X, pd.Series):
            data = X.dropna()
            return data.to_numpy().reshape(-1, 1), data.index
        elif isinstance(X, pd.DataFrame):
            data = X.dropna()
            return data.to_numpy(), data.index
        arr = np.asarray(X, dtype=float)
        arr = arr.reshape(-1, 1) if arr.ndim == 1 else arr
        return arr, None

    def fit(self, X) -> HMMClusteringResult:
        obs, index = self._prepare_observations(X)
        model = GaussianHMM(n_components=self.n_states, covariance_type=self.covariance_type, random_state=self.random_state)
        model.fit(obs)
        hidden_states = model.predict(obs)

        means, stds, periods = [], [], []
        for i in range(self.n_states):
            state_data = obs[hidden_states == i]
            means.append(np.mean(state_data))
            stds.append(np.std(state_data))
            periods.append(len(state_data))

        if index is not None:
            label_output: Union[np.ndarray, pd.Series] = pd.Series(hidden_states, index=index)
        else:
            label_output = hidden_states

        self.labels_ = hidden_states
        self.model_ = model

        return HMMClusteringResult(labels=label_output, means=means, stds=stds, periods=periods)

    def predict(self, X) -> Union[np.ndarray, pd.Series]:
        if self.model_ is None:
            raise RuntimeError("Model not fitted yet.")
        obs, index = self._prepare_observations(X)
        preds = self.model_.predict(obs)
        if index is not None:
            return pd.Series(preds, index=index)
        return preds
