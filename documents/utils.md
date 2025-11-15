# `src/utils.py`

Helper functions for working with segmented time series, visualising regimes, and pulling Yahoo! Finance reference data.

- `segment_time_series(series, window, step)`: accepts a pandas Series and returns a Series of numpy-array segments indexed by the segment end timestamp (used directly by the pipeline and WK-means).
- `segment_stats(segments, use_std=True)`: computes mean and variance (or std) per segment.
- `scatter_mean_variance(...)`: plots segments in mean–variance space with optional centroids, using Tol's academic palette (blue/red/green) for the first three regimes (extra regimes fall back to matplotlib colors).
- `plot_regimes_over_price(prices, labels, title, highlight_clusters=None, highlight_min_width=1)`: accepts price data (Series or array) and a label Series/array. Labels are asof-aligned to the price index and the function draws a single line whose color switches when regimes change (Tol palette). Pass `highlight_clusters` (iterable of regime ids) to shade the corresponding contiguous spans, and control the span size with `highlight_min_width`.
- `download_prices(tickers, start, end, field="Close")`: convenience wrapper around `yfinance.download` that handles multi-index outputs, strips timezones, enforces ticker coverage, and caches results to `data/stocks.csv`.
- `download_market_caps(tickers, start, end, prices=None)`: obtains historical shares outstanding (via `get_shares_full`, cached under `data/share_counts_full.csv`), aligns them to the daily price index, and returns the resulting market-cap time series (also cached at `data/market_cap.csv`).

These helpers are shared by the clustering demo notebooks and the trading pipeline.
