# `src/utils.py`

Helper functions for working with segmented time series and visualising regimes.

- `segment_time_series(series, window, step)`: accepts a pandas Series and returns a Series of numpy-array segments indexed by the segment end timestamp (used directly by the pipeline and WK-means).
- `segment_stats(segments, use_std=True)`: computes mean and variance (or std) per segment.
- `scatter_mean_variance(...)`: plots segments in mean–variance space with optional centroids, using Tol's academic palette (blue/red/green) for the first three regimes (extra regimes fall back to matplotlib colors).
- `plot_regimes_over_price(prices, labels, title, highlight_clusters=None, highlight_min_width=1)`: accepts price data (Series or array) and a label Series/array. Labels are asof-aligned to the price index and the function draws a single line whose color switches when regimes change (Tol palette). Pass `highlight_clusters` (iterable of regime ids) to shade the corresponding contiguous spans, and control the span size with `highlight_min_width`.

These helpers are shared by the clustering demo notebooks and the trading pipeline.
