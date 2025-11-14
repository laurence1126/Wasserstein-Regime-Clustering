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
    prices: np.ndarray,
    segments: Optional[List[np.ndarray]],
    labels: np.ndarray,
    window: int,
    step: int,
    title="Market Regimes",
    highlight_clusters: Optional[Union[int, Iterable[int], str]] = None,
    highlight_color: str = "red",
    highlight_alpha: float = 0.15,
    highlight_window: Optional[int] = None,
    highlight_score_threshold: Optional[float] = None,
    highlight_score_window: int = 5,
    highlight_min_width: int = 1,
):
    """
    Plot price series with coloring by WK-means cluster membership and optional rectangular highlights.

    Parameters
    ----------
    prices : array-like
        Full price series.
    segments : list of arrays or None
        Segmented returns used in WK-means. If None, `labels` are assumed to align directly with the price series (e.g., HMM labels).
    labels : array-like
        Cluster assignment (per segment or per observation).
    window : int
        Segment length (same as used in segmenting).
    step : int
        Step size between segments.
    title : str
        Plot title.
    times : array-like or None
        Optional time axis for the price series. Defaults to index positions.
    highlight_clusters : int | Iterable[int] | str | None
        When provided, draw translucent rectangles over periods dominated by the specified clusters.
        Use an int for a single cluster, an iterable for multiple clusters, or ``\"all\"`` to highlight every regime.
    highlight_color : str
        Color for the highlighting rectangles (default ``"red"``).
    highlight_alpha : float
        Opacity for the highlight rectangles (default 0.15).
    highlight_window : int | None
        Optional override for the width (in samples) of highlighted windows when ``segments`` are provided.
        Defaults to the clustering window length.
    highlight_score_threshold : float | None
        If set, compute a rolling score for the requested clusters and only highlight regions whose smoothed score
        exceeds this threshold. Requires ``highlight_clusters``.
    highlight_score_window : int
        Window length for the rolling mean applied to the scores (>=1). Only used when ``highlight_score_threshold`` is set.
    highlight_min_width : int
        Minimum number of samples a highlighted span must cover; narrower regions are skipped (default 1 sample).
    """

    if isinstance(prices, pd.Series):
        times = prices.index
    else:
        times = np.arange(len(prices))
    prices = np.asarray(prices)

    if isinstance(labels, pd.Series):
        labels = labels.values

    _, ax = plt.subplots(figsize=(12, 5))

    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab10", len(unique_labels))

    colors = np.zeros(len(prices))
    counts = np.zeros(len(prices))

    effective_highlight_window = None
    if segments is not None and len(segments):
        effective_highlight_window = highlight_window or window
        if effective_highlight_window <= 0:
            raise ValueError("highlight_window must be positive.")
        for idx, _ in enumerate(segments):
            start = idx * step
            end = start + window
            if end >= len(prices):
                break
            colors[start:end] += labels[idx]
            counts[start:end] += 1

        avg_labels = np.divide(colors, counts, out=np.zeros_like(colors), where=counts > 0)
        avg_labels = np.round(avg_labels).astype(int)
        scatter = ax.scatter(times, prices, c=avg_labels, cmap=cmap, s=10, alpha=0.85)
        point_labels = avg_labels
        point_times = times
        coverage_mask = counts > 0
    else:
        scatter = ax.scatter(times[1:], prices[1:], c=labels, cmap=cmap, s=10, alpha=0.85)
        point_labels = np.asarray(labels, dtype=int)
        point_times = np.asarray(times[1:])
        coverage_mask = np.ones_like(point_labels, dtype=bool)

    def _prepare_target_set(target, valid_labels):
        if target is None:
            return set()
        if isinstance(target, str):
            if target.lower() == "all":
                return set(valid_labels)
            return {int(target)}
        try:
            iter(target)  # type: ignore[arg-type]
            targets = target
        except TypeError:
            targets = [target]
        return {int(t) for t in targets}

    def _contiguous_spans(label_arr, mask, targets):
        spans = []
        start = None
        for idx, (lab, valid) in enumerate(zip(label_arr, mask)):
            if valid and lab in targets:
                if start is None:
                    start = idx
            else:
                if start is not None:
                    spans.append((start, idx - 1))
                    start = None
        if start is not None:
            spans.append((start, len(label_arr) - 1))
        return spans

    def _span_end(idx, t_axis):
        if idx < len(t_axis) - 1:
            return t_axis[idx + 1]
        if len(t_axis) > 1:
            delta = t_axis[-1] - t_axis[-2]
            return t_axis[-1] + delta
        return t_axis[-1]

    def _mask_to_spans(mask):
        spans = []
        start = None
        for idx, flag in enumerate(mask):
            if flag:
                if start is None:
                    start = idx
            elif start is not None:
                spans.append((start, idx - 1))
                start = None
        if start is not None:
            spans.append((start, len(mask) - 1))
        return spans

    def _rolling_mean(values, window_len):
        if window_len <= 1:
            return values
        values = np.asarray(values, dtype=float)
        mask = np.isfinite(values)
        filled = np.nan_to_num(values, nan=0.0)
        kernel = np.ones(window_len, dtype=float)
        summed = np.convolve(filled, kernel, mode="same")
        counts_conv = np.convolve(mask.astype(float), kernel, mode="same")
        out = np.full_like(values, np.nan, dtype=float)
        np.divide(summed, counts_conv, out=out, where=counts_conv > 0)
        return out

    def _merge_intervals(intervals):
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda pair: pair[0])
        merged = [intervals[0]]
        for start, end in intervals[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged

    if highlight_score_threshold is not None and highlight_clusters is None:
        raise ValueError("highlight_score_threshold requires highlight_clusters to be set.")
    if highlight_score_window < 1:
        raise ValueError("highlight_score_window must be >= 1.")
    if highlight_min_width < 1:
        raise ValueError("highlight_min_width must be >= 1.")

    if highlight_clusters is not None:
        valid_label_values = np.unique(point_labels[coverage_mask])
        targets = _prepare_target_set(highlight_clusters, valid_label_values)
        if targets:
            spans = []
            span_times = times if len(segments) else point_times
            if highlight_score_threshold is not None:
                if len(segments):
                    target_counts = np.zeros(len(prices), dtype=float)
                    max_idx = len(prices)
                    for idx, lab in enumerate(labels):
                        if lab in targets:
                            start_idx = idx * step
                            end_idx = min(start_idx + window, max_idx)
                            target_counts[start_idx:end_idx] += 1
                    total_counts = counts
                    scores = np.full(len(prices), np.nan, dtype=float)
                    np.divide(target_counts, total_counts, out=scores, where=total_counts > 0)
                    scores = _rolling_mean(scores, highlight_score_window)
                    mask = np.isfinite(scores) & (scores >= highlight_score_threshold)
                    spans = _mask_to_spans(mask)
                else:
                    scores = np.array([1.0 if lab in targets else 0.0 for lab in point_labels], dtype=float)
                    scores = _rolling_mean(scores, highlight_score_window)
                    mask = np.isfinite(scores) & (scores >= highlight_score_threshold)
                    spans = _mask_to_spans(mask)
            else:
                if len(segments):
                    intervals = []
                    max_idx = len(prices) - 1
                    for idx, lab in enumerate(labels):
                        if lab in targets:
                            start_idx = idx * step
                            end_idx = min(start_idx + effective_highlight_window, max_idx)
                            intervals.append((start_idx, end_idx))
                    spans = _merge_intervals(intervals)
                else:
                    spans = _contiguous_spans(point_labels, coverage_mask, targets)
                span_times = times if len(segments) else point_times

            for start_idx, end_idx in spans:
                if (end_idx - start_idx + 1) < highlight_min_width:
                    continue
                if start_idx >= len(span_times):
                    continue
                clipped_end = min(end_idx, len(span_times) - 1)
                x0 = span_times[start_idx]
                x1 = _span_end(clipped_end, span_times)
                ax.axvspan(
                    x0,
                    x1,
                    facecolor=highlight_color,
                    edgecolor=highlight_color,
                    alpha=highlight_alpha,
                    zorder=0,
                    lw=1.0,
                )

    ax.set_title(title)
    if isinstance(times, pd.DatetimeIndex):
        ax.set_xlabel("Time")
    else:
        ax.set_xlabel("Time (index)")
    ax.set_ylabel("Price")
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(len(unique_labels)))
    cbar.set_label("Cluster")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()
