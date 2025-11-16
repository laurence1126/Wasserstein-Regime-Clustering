import numpy as np
import matplotlib.pyplot as plt

from .constants import CLUSTER_PALETTE


class MMDCalculator:
    """
    MMD^2 on equal-length segments with Gaussian RBF kernel.
    - Median heuristic for bandwidth (default).
    - Biased (V-statistic) or unbiased (U-statistic).
    - Supports: between-cluster MMD^2, within-cluster MMD^2 (by random split),
      and convenience plotting for two methods (e.g., Wasserstein vs Moments).
    """

    def __init__(self, sigma=None, unbiased=False, random_state=None):
        self.sigma = sigma
        self.unbiased = unbiased
        self.rng = np.random.default_rng(random_state)

    # ---------- utilities ----------
    @staticmethod
    def _as_2d(X):
        X = np.asarray(X, dtype=float)
        return X if X.ndim == 2 else X.reshape(1, -1)

    @staticmethod
    def _sq_dists(A, B):
        A2 = np.sum(A * A, axis=1, keepdims=True)
        B2 = np.sum(B * B, axis=1, keepdims=True).T
        return A2 + B2 - 2.0 * (A @ B.T)

    def _rbf(self, A, B, sigma):
        return np.exp(-self._sq_dists(A, B) / (2.0 * sigma * sigma))

    def _median_sigma(self, X, Y=None):
        Z = X if Y is None else np.vstack([X, Y])
        if Z.shape[0] < 3:
            return 1.0
        d2 = self._sq_dists(Z, Z)
        tri = d2[np.triu_indices_from(d2, k=1)]
        med = np.median(np.sqrt(np.maximum(tri, 0.0)))
        return med if med > 0 else 1.0

    # ---------- core MMD^2 ----------
    def mmd2(self, X, Y, sigma=None):
        X = self._as_2d(X)
        Y = self._as_2d(Y)
        sig = sigma if sigma is not None else (self.sigma or self._median_sigma(X, Y))

        Kxx = self._rbf(X, X, sig)
        Kyy = self._rbf(Y, Y, sig)
        Kxy = self._rbf(X, Y, sig)

        nx, ny = X.shape[0], Y.shape[0]
        if self.unbiased:
            np.fill_diagonal(Kxx, 0.0)
            np.fill_diagonal(Kyy, 0.0)
            term_xx = Kxx.sum() / (nx * (nx - 1)) if nx > 1 else 0.0
            term_yy = Kyy.sum() / (ny * (ny - 1)) if ny > 1 else 0.0
        else:
            term_xx = Kxx.sum() / (nx * nx)
            term_yy = Kyy.sum() / (ny * ny)

        term_xy = Kxy.sum() / (nx * ny)
        return float(term_xx + term_yy - 2.0 * term_xy)

    # ---------- BETWEEN-cluster bootstrap ----------
    def bootstrap_between(self, X, Y, B=2000, m_per_group=40, replace=True):
        X = self._as_2d(X)
        Y = self._as_2d(Y)
        nX, nY = X.shape[0], Y.shape[0]
        m = min(m_per_group, nX, nY)
        out = np.empty(B, dtype=float)
        for b in range(B):
            i = self.rng.choice(nX, size=m, replace=replace)
            j = self.rng.choice(nY, size=m, replace=replace)
            out[b] = self.mmd2(X[i], Y[j])
        return out

    # ---------- WITHIN-cluster bootstrap (random split into two halves) ----------
    def bootstrap_within(self, X, B=2000, m_per_half=40, replace=False):
        """
        Within a single cluster X (shape N x L), repeatedly:
        - sample 2*m_per_half indices,
        - split into A and B (m each),
        - compute MMD^2(A, B).
        """
        X = self._as_2d(X)
        N = X.shape[0]
        m = min(m_per_half, N // 2) if not replace else m_per_half
        if m == 0:
            raise ValueError("Not enough samples in cluster for within-cluster bootstrap.")
        out = np.empty(B, dtype=float)
        for b in range(B):
            if replace:
                idx = self.rng.choice(N, size=2 * m, replace=True)
                A = X[idx[:m]]
                Bset = X[idx[m:]]
            else:
                idx = self.rng.choice(N, size=2 * m, replace=False)
                self.rng.shuffle(idx)
                A = X[idx[:m]]
                Bset = X[idx[m : 2 * m]]
            out[b] = self.mmd2(A, Bset)
        return out

    # ---------- helpers to pick clusters ----------
    @staticmethod
    def _largest_two_clusters(segments, labels):
        segments = np.asarray(segments)
        labels = np.asarray(labels, dtype=int)
        uniq, counts = np.unique(labels, return_counts=True)
        order = np.argsort(-counts)
        labs = uniq[order[:2]]
        return [segments[labels == labs[0]], segments[labels == labs[1]]], labs

    # ---------- PLOTS ----------
    def compare_two_clusterings_hist(
        self,
        segments,
        labels_a,
        labels_b,
        B=4000,
        m_per_group=40,
        bins=120,
        title=r"Between-cluster MMD$^2$",
        names=("Wasserstein", "Moments"),
        figsize=(10, 5),
        alpha_a=0.7,
        alpha_b=0.6,
    ):
        """
        segments: list/array of equal-length windows (shape N x L)
        labels_a: cluster labels for 'method A' (assumes binary: two regimes 0/1)
        labels_b: cluster labels for 'method B' (assumes binary)
        Returns the two bootstrap arrays and the figure.
        """
        S = self._as_2d(segments)
        lab_a = np.asarray(labels_a, dtype=int)
        lab_b = np.asarray(labels_b, dtype=int)

        # assume two regimes; if more exist, take the two most populated
        def two_biggest_pairs(labels):
            uniq, counts = np.unique(labels, return_counts=True)
            order = np.argsort(-counts)
            c0, c1 = uniq[order[:2]]
            X = S[labels == c0]
            Y = S[labels == c1]
            return X, Y

        XA, YA = two_biggest_pairs(lab_a)
        XB, YB = two_biggest_pairs(lab_b)

        dist_a = self.bootstrap_between(XA, YA, B=B, m_per_group=m_per_group)
        dist_b = self.bootstrap_between(XB, YB, B=B, m_per_group=m_per_group)

        fig, ax = plt.subplots(figsize=figsize)
        fallback = plt.get_cmap("tab10")
        colors = [CLUSTER_PALETTE.get(0, fallback(0)), CLUSTER_PALETTE.get(2, fallback(1))]
        ax.hist(dist_a, bins=bins, density=True, alpha=alpha_a, label=names[0], color=colors[0])
        ax.hist(dist_b, bins=bins, density=True, alpha=alpha_b, label=names[1], color=colors[1])
        ax.set_xlabel(r"$\mathrm{MMD}_b^2$")
        ax.set_ylabel("density")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.25)
        return dist_a, dist_b, fig

    def plot_within_two_methods(
        self,
        segments,
        labels_a,  # e.g., Wasserstein labels
        labels_b,  # e.g., Moments labels
        B=4000,
        m_per_half=40,
        bins=120,
        names=("Wasserstein", "Moments"),
        figsize=(11, 4.5),
        sharey=True,
    ):
        """
        Reproduces the 'within-cluster MMD^2' comparison figure (two panels).
        For each method, take its two largest clusters; for each cluster, build
        a within-cluster MMD^2 bootstrap distribution and plot overlays.
        Left panel = cluster #1 (largest), Right panel = cluster #2 (2nd largest).
        """
        S = self._as_2d(segments)
        (XA, YA), _ = self._largest_two_clusters(S, labels_a)
        (XB, YB), _ = self._largest_two_clusters(S, labels_b)

        # Build within-cluster distributions
        A1 = self.bootstrap_within(XA, B=B, m_per_half=m_per_half)
        A2 = self.bootstrap_within(YA, B=B, m_per_half=m_per_half)
        B1 = self.bootstrap_within(XB, B=B, m_per_half=m_per_half)
        B2 = self.bootstrap_within(YB, B=B, m_per_half=m_per_half)

        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=sharey)
        fallback = plt.get_cmap("tab10")
        colors = [CLUSTER_PALETTE.get(0, fallback(0)), CLUSTER_PALETTE.get(2, fallback(1))]
        # Panel 1: biggest cluster per method
        axes[0].hist(A1, bins=bins, density=True, alpha=0.7, label=names[0], color=colors[0])
        axes[0].hist(B1, bins=bins, density=True, alpha=0.6, label=names[1], color=colors[1])
        axes[0].set_title("Within-cluster MMD$^2$ (largest cluster)")
        axes[0].set_xlabel(r"$\mathrm{MMD}_b^2$")
        axes[0].set_ylabel("density")
        axes[0].legend()
        axes[0].grid(alpha=0.25)

        # Panel 2: second-largest cluster per method
        axes[1].hist(A2, bins=bins, density=True, alpha=0.7, label=names[0], color=colors[0])
        axes[1].hist(B2, bins=bins, density=True, alpha=0.6, label=names[1], color=colors[1])
        axes[1].set_title("Within-cluster MMD$^2$ (2nd largest cluster)")
        axes[1].set_xlabel(r"$\mathrm{MMD}_b^2$")
        axes[1].legend()
        axes[1].grid(alpha=0.25)

        fig.tight_layout()
        return (A1, A2, B1, B2), fig
