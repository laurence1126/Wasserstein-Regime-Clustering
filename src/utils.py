import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterable, List, Optional, Union


def segment_time_series(series: pd.Series, window: int, step: int) -> pd.Series:
    """
    Segment a 1D time series into overlapping windows.

    Parameters
    ----------
    series : pandas.Series
        Input time series (e.g., log-returns) with a datetime-like index.
    window : int
        Length of each segment (h1).
    step : int
        Step size (overlap offset, h2).
        - step < window => overlapping segments
        - step = window => disjoint segments
    Returns
    -------
    pandas.Series
        Series whose values are numpy arrays (segments) and index is the
        timestamp corresponding to the end of each segment.
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    series = series.astype(float)
    n = len(series)
    windows: List[np.ndarray] = []
    idx: List[pd.Timestamp] = []
    for start in range(0, n - window + 1, step):
        end = start + window
        segment = series.iloc[start:end].to_numpy(copy=True)
        windows.append(segment)
        idx.append(series.index[end - 1])
    return pd.Series(windows, index=pd.Index(idx, name="segment_end"))


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

    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab10", len(unique_labels))
    palette = {0: "#4477AA", 1: "#228833", 2: "#EE6677", 3: "#CCBB44", 4: "#66CCEE", 5: "#AA3377", 6: "#BBBBBB"}
    colors = {lab: palette.get(lab, cmap(i)) for i, lab in enumerate(unique_labels)}

    plt.figure(figsize=(8, 6))
    for lab in unique_labels:
        mask = labels == lab
        plt.scatter(v_or_s[mask], means[mask], s=s, alpha=alpha, color=colors[lab], label=f"Cluster {lab}")

    # Optional centroid overlay (in mean–variance space, not Wasserstein barycenters)
    if show_centroids:
        for lab in unique_labels:
            mask = labels == lab
            if np.any(mask):
                c_mean = means[mask].mean()
                c_var = v_or_s[mask].mean()
                plt.scatter([c_var], [c_mean], s=160, edgecolor="black", linewidth=1.2, color=colors[lab], marker="X", zorder=5)

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
    prices,
    labels,
    title="Market Regimes",
    highlight_clusters: Optional[Iterable[int]] = None,
    highlight_min_width: int = 1,
):
    """Plot price history with contiguous line segments coloured by regime labels."""

    if isinstance(prices, pd.Series):
        price_series = prices.sort_index().astype(float)
    else:
        arr = np.asarray(prices, dtype=float)
        price_series = pd.Series(arr, index=pd.RangeIndex(len(arr)))

    if isinstance(labels, pd.Series):
        label_series = labels.sort_index().dropna().astype(int)
    else:
        label_series = pd.Series(labels, index=price_series.index[: len(labels)]).astype(int)

    aligned = pd.merge_asof(
        price_series.to_frame("price"),
        label_series.to_frame("label"),
        left_index=True,
        right_index=True,
        direction="backward",
    ).dropna()
    if aligned.empty:
        raise ValueError("No overlapping timestamps between prices and labels")
    point_times = aligned.index.to_numpy()
    point_prices = aligned["price"].to_numpy()
    point_labels = aligned["label"].to_numpy(dtype=int)

    _, ax = plt.subplots(figsize=(12, 5))
    unique_labels = np.unique(point_labels)
    cmap = plt.get_cmap("tab10", len(unique_labels))

    palette = {0: "#4477AA", 1: "#228833", 2: "#EE6677", 3: "#CCBB44", 4: "#66CCEE", 5: "#AA3377", 6: "#BBBBBB"}
    label_colors = {lab: palette.get(lab, cmap(i)) for i, lab in enumerate(unique_labels)}
    plotted = set()
    segments = []
    start_idx = 0
    current_label = point_labels[0]
    for idx in range(1, len(point_labels)):
        if point_labels[idx] != current_label:
            segments.append((current_label, point_times[start_idx : idx + 1], point_prices[start_idx : idx + 1]))
            start_idx = idx
            current_label = point_labels[idx]
    segments.append((current_label, point_times[start_idx:], point_prices[start_idx:]))

    highlight_set = set(highlight_clusters or [])
    for lab, times_seg, prices_seg in segments:
        color = label_colors[lab]
        label = f"Cluster {lab}" if lab not in plotted else None
        plotted.add(lab)
        ax.plot(times_seg, prices_seg, color=color, label=label)
        if lab in highlight_set and len(times_seg) >= highlight_min_width:
            ax.axvspan(times_seg[0], times_seg[-1], color="red", alpha=0.1, zorder=0)

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()
