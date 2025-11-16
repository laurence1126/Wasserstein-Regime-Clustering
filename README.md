# Wasserstein Regime Clustering

Toolkit for regime detection on SPX intraday returns using Wasserstein K-means plus an illustrative growth/defensive rotation strategy. Code lives in `src/` while exploratory notebooks are in `jupyter/`.

## Repository Layout

```
├── src/
│   ├── clustering_methods.py       # Wasserstein & moment K-means utilities
│   ├── clustering_eval.py          # MMD calculator + clustering metrics
│   ├── utils.py                    # segmentation, plotting, Yahoo helpers
│   ├── jump_diffusion.py           # Merton jump-diffusion simulators + benchmark
│   ├── constants.py                # shared palette & globals
│   └── regime_trading_pipeline.py  # RegimeRotationStrategy & grid search
├── jupyter/
│   ├── clustering_examples.ipynb        # walkthrough of segmentation + clustering
│   ├── clustering_eval_examples.ipynb   # standalone evaluation notebook
│   ├── jump_diffusion_compare.ipynb     # jump-diffusion benchmark + visualization notebook
│   └── trading.ipynb                    # end-to-end strategy analysis & plots
├── documents/
│   ├── clustering_method.md        # auto-generated module docs
│   ├── clustering_eval.md
│   ├── utils.md
│   └── regime_rotation.md
├── data/
│   ├── SPX_hourly.csv              # hourly SPX prices (signal driver)
│   ├── stocks.csv                  # cached Yahoo Finance closes
│   └── market_cap.csv              # derived market-cap time series
└── README.md
```

## Python Modules

### `src/clustering_methods.py`

- `WassersteinKMeans`: supports warm-starts (passing previous centroids) and accepts either a list of numpy arrays or a pandas Series of segments. `predict` returns a Series if given Series inputs, preserving timestamps.
- `MomentKMeans`: classic K-means on raw moments with k-means++ initialization.

- `segment_time_series(series, window, step)`: slices a pandas Series into overlapping windows and returns a Series with segment-end timestamps.
- `segment_stats`, `scatter_mean_variance`, `plot_regimes_over_price`: statistics/visualization helpers. Plots use a professional Tol palette (blue/red/green) and can highlight specific regimes (e.g., `plot_regimes_over_price(..., highlight_clusters=[0,2], highlight_min_width=5)`); regime legends are sorted numerically (Cluster 0, 1, 2, …) regardless of plotting order.
- `load_signal(signal_path, start_date=None, end_date=None)`: convenience parser for the semicolon-delimited SPX intraday CSV—builds a datetime index, computes returns, and clips to the requested window.
- `download_prices(tickers, start, end, field="Close")`: Yahoo! Finance downloader with csv caching and robust multi-index handling.
- `download_market_caps(tickers, start, end, prices=None)`: stitches shares outstanding histories (via `get_shares_full` cached under `data/share_counts_full.csv`) with the downloaded prices to form a market-cap time series used for weighting schemes.

### `src/clustering_eval.py`

- `MMDCalculator`: RBF MMD implementation with between/within bootstrap routines and comparison plots used in the notebooks.
- `ClusteringMetrics`: Davies-Bouldin, Dunn, and (α-)silhouette metrics mirroring the evaluation section of the paper.

### `src/constants.py`

- `CLUSTER_PALETTE`: single source of truth for regime/cluster colors used across plotting utilities and notebooks. Import via `from src import CLUSTER_PALETTE` to keep figures consistent.

- `RegimeRotationStrategy`: class encapsulating signal preparation, rolling WK-means fitting (with hot start), daily return construction (calling the shared `download_prices`/`download_market_caps` helpers), and backtesting for a growth vs. defensive rotation. Supports both equal-weight and market-cap-weighted legs via the `weighting` argument. `StrategyResult` includes strategy curve plus benchmark curves (SPY + each leg/allocation under both schemes when available).
- `grid_search_regimes`: iterates across window/step/refit grids and reports metrics, with a `rich` progress bar.

### `src/jump_diffusion.py`

- `JumpDiffusionParams`: typed container for (μ, σ, λ, γ, δ).
- `MertonJumpDiffusion`: class with `simulate_path` and `log_return_moments(dt)` for single-regime paths.
- `RegimeSwitchingMerton`: bull/bear simulator with jittered regime windows (both change-point locations and optional per-window length perturbations via `length_jitter`). Call `simulate(...)` to reproduce Figure 9.
- `MertonBenchmark`: orchestrates repeated simulations + clustering (Wasserstein vs. Moments by default) to produce the accuracy table (Section 3.3.2). Call `run(return_details=True)` to also retrieve the simulated price path, segmented windows, true regimes, and per-algorithm label Series—handy for piping into `plot_regimes_over_price`, `scatter_mean_variance`, etc. Optional helpers (`plot_accuracy_bars`, `plot_sample_path`) visualize aggregate accuracy or a representative run. `run_merton_benchmark` remains as a helper that instantiates the class.
- Compatibility wrappers `simulate_merton_jump_diffusion` / `simulate_merton_jump_regimes` are provided for notebooks/scripts that previously imported them from `utils`.

## Notebooks (`jupyter/`)

- **`clustering_examples.ipynb`** – walkthrough of segmenting SPX returns, fitting Wasserstein vs. moment K-means, visualizing regimes, and comparing clustering quality via MMD.
- **`clustering_eval_examples.ipynb`** – dedicated evaluation notebook showing how to run `ClusteringMetrics`/MMD comparisons (Davies–Bouldin, Dunn, silhouette, bootstrapped MMD) on WK-means vs. moment K-means outputs.
- **`jump_diffusion_compare.ipynb`** – generates the synthetic Merton benchmark table via `MertonBenchmark.run(return_details=True)` and pipes the stored price/segment/prediction Series into `plot_regimes_over_price` and `scatter_mean_variance` to contrast the true regimes with Wasserstein vs. Moment assignments.
- **`trading.ipynb`** – uses `RegimeRotationStrategy` to fit regimes, run backtests with different allocation maps, plot equity curves (with drawdown highlights) and generate figures for reporting.

## Usage

1. Install dependencies:
   ```bash
   pip3 install -r requirements.txt  # ensure pandas, numpy, matplotlib, rich, yfinance
   ```
2. Place `SPX_hourly.csv` in `data/` (hourly SPX prices separated by semicolons).
3. Run the pipeline:

   ```bash
   python3 -m src.regime_trading_pipeline
   ```

   or import `RegimeRotationStrategy` inside notebooks/scripts.

4. Optional hyper-parameter tuning:
   ```python
   from src.regime_trading_pipeline import grid_search_regimes
   grid = grid_search_regimes(growth, defensive, windows=(240, 300), steps=(6, 12), refits=(24, 48))
   print(grid.head())
   ```

## Documentation

Auto-generated markdown files reside in `documents/` summarizing each module (`clusterings.md`, `utils.md`, `MMD.md`, `regime_rotation.md`).

Feel free to explore the notebooks for visual examples and adapt the strategy class to your own baskets of tickers.
