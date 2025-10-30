import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional


def segment_time_series(series: np.ndarray, window: int, step: int) -> List[np.ndarray]:
    """
    Segment a 1D time series into overlapping windows.

    Parameters
    ----------
    series : array-like
        Input time series (e.g., log-returns).
    window : int
        Length of each segment (h1).
    step : int
        Step size (overlap offset, h2).
        - step < window => overlapping segments
        - step = window => disjoint segments
    Returns
    -------
    segments : list of np.ndarray
        List of segments, each of length `window`.
    """
    series = np.asarray(series, dtype=float).ravel()
    n = len(series)
    segments = []
    for start in range(0, n - window + 1, step):
        segment = series[start : start + window]
        segments.append(segment)
    return segments


def segment_stats(segments: List[np.ndarray], use_std=True):
    """
    Compute (mean, variance) or (mean, std) for each segment.
    Returns two numpy arrays of shape (n_segments,).
    """
    segs = [np.asarray(s, dtype=float).ravel() for s in segments]
    means = np.array([s.mean() for s in segs])
    variances = np.array([s.var(ddof=1) for s in segs])
    if use_std:
        return means, np.sqrt(variances)
    return means, variances


def scatter_mean_variance(segments, labels, title="Segments in Mean–Variance Space", use_std=True, alpha=0.7, s=18, show_centroids=True, legend=True):
    """
    Scatter plot of segments in (mean, variance) or (mean, std) space, colored by cluster labels.

    Parameters
    ----------
    segments : list[np.ndarray]
        List of return windows (equal length recommended).
    labels : array-like of int
        Cluster assignment for each segment (0..K-1).
    title : str
        Plot title.
    use_std : bool
        If True, y-axis is standard deviation; else variance.
    alpha : float
        Point transparency.
    s : int
        Marker size.
    show_centroids : bool
        If True, overlay cluster centroids computed from (mean, var/std) of members.
    legend : bool
        If True, show legend for clusters.
    """
    labels = np.asarray(labels, dtype=int)
    means, v_or_s = segment_stats(segments, use_std=use_std)

    K = int(labels.max()) + 1
    cmap = plt.get_cmap("tab10", K)

    plt.figure(figsize=(8, 6))
    for k in range(K):
        mask = labels == k
        plt.scatter(v_or_s[mask], means[mask], s=s, alpha=alpha, color=cmap(k), label=f"Cluster {k}")

    # Optional centroid overlay (in mean–variance space, not Wasserstein barycenters)
    if show_centroids:
        for k in range(K):
            mask = labels == k
            if np.any(mask):
                c_mean = means[mask].mean()
                c_var = v_or_s[mask].mean()
                plt.scatter([c_var], [c_mean], s=160, edgecolor="black", linewidth=1.2, color=cmap(k), marker="X", zorder=5)

    xlab = "Std. Dev." if use_std else "Variance"
    plt.xlabel(xlab)
    plt.ylabel("Mean")
    plt.title(title)
    if legend:
        plt.legend(frameon=False)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


def plot_regimes_over_price(
    prices: np.ndarray, segments: Optional[List[np.ndarray]], labels: np.ndarray, window: int, step: int, title="Market Regimes", times=None
):
    """
    Plot price series with coloring by WK-means cluster membership.

    Parameters
    ----------
    prices : array-like
        Full price series.
    segments : list of arrays
        Segmented returns used in WK-means.
    labels : array-like
        Cluster assignment for each segment.
    window : int
        Segment length (same as used in segmenting).
    step : int
        Step size between segments.
    title : str
        Plot title.
    """
    prices = np.asarray(prices)
    if times is None:
        times = np.arange(len(prices))
    fig, ax = plt.subplots(figsize=(12, 5))

    # Default color map: cluster 0 = green, cluster 1 = red, etc.
    cmap = plt.get_cmap("tab10", len(set(labels)))

    # Assign each segment's cluster to its covered price indices
    colors = np.zeros(len(prices))
    counts = np.zeros(len(prices))

    if segments:
        for idx, seg in enumerate(segments):
            start = idx * step
            end = start + window
            if end >= len(prices):
                break
            colors[start:end] += labels[idx]
            counts[start:end] += 1

        # Average cluster assignment where segments overlap
        avg_labels = np.divide(colors, counts, out=np.zeros_like(colors), where=counts > 0)
        avg_labels = np.round(avg_labels).astype(int)

        # Plot price series, coloring by cluster
        scatter = ax.scatter(times, prices, c=avg_labels, cmap=cmap, s=10, alpha=0.8)
    else:
        scatter = ax.scatter(times[1:], prices[1:], c=labels, cmap=cmap, s=10, alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("Time (index)")
    ax.set_ylabel("Price")
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(len(set(labels))))
    cbar.set_label("Cluster")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()
