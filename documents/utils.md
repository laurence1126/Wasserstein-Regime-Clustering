# `src/utils.py`

Helper functions for working with segmented time series and visualising regimes.

- `segment_time_series(series, window, step)`: accepts a pandas Series and returns a Series of numpy-array segments indexed by the segment end timestamp (used directly by the pipeline and WK-means).
- `segment_stats(segments, use_std=True)`: computes mean and variance (or std) per segment.
- `scatter_mean_variance(...)`: plots segments in mean–variance space with optional centroids.
- `plot_regimes_over_price(...)`: accepts price data (Series or array) and regime labels. When labels are provided as a Series with timestamps they are asof-aligned with prices, so the plot shows only time points where labels exist. Highlighting works with both Series-based labels and legacy segment-based inputs.

These helpers are shared by the clustering demo notebooks and the trading pipeline.
