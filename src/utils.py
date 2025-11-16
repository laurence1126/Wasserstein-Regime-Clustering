import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .constants import CLUSTER_PALETTE

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


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
    colors = {lab: CLUSTER_PALETTE.get(lab, cmap(i)) for i, lab in enumerate(unique_labels)}

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

    label_colors = {lab: CLUSTER_PALETTE.get(lab, cmap(i)) for i, lab in enumerate(unique_labels)}
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
    merged_spans = []
    span_start = None
    span_end = None
    span_count = 0

    for lab, times_seg, prices_seg in segments:
        color = label_colors[lab]
        label = f"Cluster {lab}" if lab not in plotted else None
        plotted.add(lab)
        ax.plot(times_seg, prices_seg, color=color, label=label)
        if lab in highlight_set:
            if span_start is None:
                span_start = times_seg[0]
            span_end = times_seg[-1]
            span_count += len(times_seg)
        else:
            if span_start is not None and span_count >= highlight_min_width:
                merged_spans.append((span_start, span_end))
            span_start = None
            span_end = None
            span_count = 0

    if span_start is not None and span_count >= highlight_min_width:
        merged_spans.append((span_start, span_end))

    for start_time, end_time in merged_spans:
        ax.axvspan(start_time, end_time, color="red", alpha=0.1, zorder=0)

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    handles, labels_text = ax.get_legend_handles_labels()
    if labels_text:
        try:
            order = sorted(range(len(labels_text)), key=lambda i: int(labels_text[i].split()[-1]))
            handles = [handles[i] for i in order]
            labels_text = [labels_text[i] for i in order]
        except ValueError:
            pass
        ax.legend(handles, labels_text, loc="upper left", frameon=False)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


def load_signal(signal_path: str | Path, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    names = ["Date", "Hour", "Open", "High", "Low", "Close", "Volume"]
    df = pd.read_csv(signal_path, sep=";", names=names)
    df["timestamp"] = pd.to_datetime(df["Date"] + " " + df["Hour"], format="%d/%m/%Y %H:%M", dayfirst=True)
    df = df.drop(columns=["Date", "Hour"]).set_index("timestamp").sort_index()
    df["Return"] = df["Close"].pct_change()

    start_date = start_date or df.index[0]
    end_date = end_date or df.index[-1]
    df = df[(end_date >= df.index) & (df.index >= start_date)]
    return df.dropna(subset=["Return"])


def download_prices(tickers: Sequence[str], start: str, end: str, field: str = "Close") -> pd.DataFrame:
    if Path.exists(Path("../data/stocks.csv")):
        df = pd.read_csv("../data/stocks.csv", index_col=0, header=0).astype(float)
        df.index = pd.to_datetime(df.index)
        missing = [t for t in tickers if t not in df.columns]
        if not missing:
            return df[(df.index >= start) & (df.index <= end)]

    if yf is None:
        raise ImportError("yfinance is required to download market data")
    data = yf.download(
        tickers=" ".join(tickers),
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )
    if data.empty:
        raise ValueError("yfinance returned no data; check tickers or date range")

    choices = {field, field.title(), field.upper()}
    if isinstance(data.columns, pd.MultiIndex):
        level1 = data.columns.get_level_values(1)
        match = next((c for c in choices if c in level1), None)
        if match is None:
            raise ValueError(f"Downloaded data lacks {field} column")
        prices = data.xs(match, level=1, axis=1)
    else:
        match = next((c for c in choices if c in data.columns), None)
        if match is None:
            raise ValueError(f"Downloaded data lacks {field} column")
        prices = data[match]

    prices = prices.ffill().dropna(how="all")
    if hasattr(prices.index, "tz") and prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise KeyError(f"Missing tickers in Yahoo download: {', '.join(missing)}")

    prices[tickers].to_csv("../data/stocks.csv")
    return prices[tickers]


import yfinance as yf
import pandas as pd


def download_market_caps(tickers: Sequence[str], start: str, end: str) -> pd.DataFrame:
    if Path("../data/market_cap.csv").exists():
        df = pd.read_csv("../data/market_cap.csv", index_col=0, header=0).astype(float)
        df.index = pd.to_datetime(df.index)
        missing = [t for t in tickers if t not in df.columns]
        if not missing:
            return df[(df.index >= start) & (df.index <= end)]

    if not Path("../data/stocks.csv").exists():
        raise RuntimeError("You need to download stock prices first!")
    else:
        price = pd.read_csv("../data/stocks.csv", index_col=0, header=0).astype(float)
        price.index = pd.to_datetime(price.index)
        missing = [t for t in tickers if t not in price.columns]
        if missing:
            raise RuntimeError("Missing tickers! You need to redownload stock prices first!")

    if yf is None:
        raise ImportError("yfinance is required to download market data")

    shares_list = []
    for t in tickers:
        tk = yf.Ticker(t)
        # get_shares_full usually returns a DataFrame with a DatetimeIndex
        shares_hist = tk.get_shares_full(start=start, end=end)

        if shares_hist is None or len(shares_hist) == 0:
            # Fallback: use current sharesOutstanding if history not available
            info = tk.info
            current_shares = info.get("sharesOutstanding")
            if current_shares is None:
                raise ValueError(f"No historical or current 'sharesOutstanding' for {t}")
            # Create a constant series over the price index
            s = pd.Series(current_shares, index=price.index, name=t)
        else:
            # If it's a DataFrame, pick the first column (commonly 'SharesOutstanding')
            if isinstance(shares_hist, pd.DataFrame):
                if shares_hist.shape[1] == 1:
                    s = shares_hist.iloc[:, 0]
                else:
                    # Try a sensible column name first, otherwise first column
                    col = "SharesOutstanding" if "SharesOutstanding" in shares_hist.columns else shares_hist.columns[0]
                    s = shares_hist[col]
            else:
                # If already a Series
                s = shares_hist
            # Make sure index is datetime and sorted
            s.index = pd.to_datetime([str(x).split(" ")[0] for x in s.index])
            s = s.groupby(s.index).last()
            s = s.sort_index()

            # Align to the daily price index with forward-fill (shares change at discrete dates)
            s = s.reindex(price.index, method="ffill")

            # In case there are leading NaNs before the first known shares value, back-fill them
            s = s.bfill()

            s.name = t

        shares_list.append(s)

    shares_df = pd.concat(shares_list, axis=1)
    shares_df = shares_df[tickers]
    market_cap_df = price * shares_df
    if "SPY" in market_cap_df.columns:
        market_cap_df.drop(columns=["SPY"])
    market_cap_df.to_csv("../data/market_cap.csv")

    return market_cap_df
