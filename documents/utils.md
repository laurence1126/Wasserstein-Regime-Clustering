# `src/utils.py`

Helper functions for working with segmented time series, visualising regimes, and pulling Yahoo! Finance reference data.

- `segment_time_series(series, window, step)`: accepts a pandas Series and returns a Series of numpy-array segments indexed by the segment end timestamp (used directly by the pipeline and WK-means).
- `segment_stats(segments, use_std=True)`: computes mean and variance (or std) per segment.
- `scatter_mean_variance(...)`: plots segments in mean–variance space with optional centroids, using Tol's academic palette (blue/red/green) for the first three regimes (extra regimes fall back to matplotlib colors).
- `plot_regimes_over_price(prices, labels, title, highlight_clusters=None, highlight_min_width=1)`: accepts price data (Series or array) and a label Series/array. Labels are asof-aligned to the price index and the function draws a single line whose color switches when regimes change (Tol palette). Pass `highlight_clusters` (iterable of regime ids) to shade the corresponding contiguous spans, control the span size with `highlight_min_width`, and note that the legend always lists regimes in ascending order (Cluster 0, 1, 2, …) independent of how series segments were processed.
- `simulate_merton_jump_diffusion(T=1.0, N=252, S0=100, mu=0.05, sigma=0.2, lam=1.0, gamma=-0.05, delta=0.1, random_state=None)`: draws a price path under the classic Merton jump-diffusion model, combining geometric Brownian motion with a Poisson jump process (jump sizes are normal with mean `gamma` and std `delta`). Returns the time grid, simulated prices, and the total number of jumps, making it easy to craft toy regimes with occasional discontinuities.
- `load_signal(signal_path, start_date=None, end_date=None)`: parses the semicolon-delimited SPX signal CSV (hourly data), builds a datetime index, computes close-to-close returns, and filters to the requested date window while dropping the initial NaN.
- `download_prices(tickers, start, end, field="Close")`: convenience wrapper around `yfinance.download` that handles multi-index outputs, strips timezones, enforces ticker coverage, and caches results to `data/stocks.csv`.
- `download_market_caps(tickers, start, end, prices=None)`: obtains historical shares outstanding (via `get_shares_full`, cached under `data/share_counts_full.csv`), aligns them to the daily price index, and returns the resulting market-cap time series (also cached at `data/market_cap.csv`).

These helpers are shared by the clustering demo notebooks and the trading pipeline.
