# Wasserstein Regime Clustering

Toolkit for regime detection on SPX intraday returns using Wasserstein K-means plus an illustrative growth/defensive rotation strategy. Code lives in `src/` while exploratory notebooks are in `jupyter/`.

## Repository Layout

```
├── src/
│   ├── clusterings.py          # Wasserstein & moment K-means utilities
│   ├── utils.py                # segmentation + plotting helpers
│   ├── MMD.py                  # MMD diagnostics/plots
│   └── regime_trading_pipeline.py  # RegimeRotationStrategy & grid search
├── jupyter/
│   ├── clustering_examples.ipynb   # exploratory WK-means / MMD visualisations
│   └── trading.ipynb               # end-to-end strategy analysis & plots
├── documents/ (auto-generated markdown docs)
├── data/
│   ├── SPX_hourly.csv  # hourly SPX prices (signal driver)
│   └── stocks.csv      # cached Yahoo Finance closes used by the pipeline
└── README.md
```

## Python Modules

### `src/clusterings.py`

- `WassersteinKMeans`: supports warm-starts (passing previous centroids) and accepts either a list of numpy arrays or a pandas Series of segments. `predict` returns a Series if given Series inputs, preserving timestamps.
- `MomentKMeans`: classic K-means on raw moments with k-means++ initialization.

### `src/utils.py`

- `segment_time_series(series, window, step)`: slices a pandas Series into overlapping windows and returns a Series with segment-end timestamps.
- `segment_stats`, `scatter_mean_variance`, `plot_regimes_over_price`: statistics/visualization helpers. Plots use a professional Tol palette (blue/red/green) and can highlight specific regimes (e.g., `plot_regimes_over_price(..., highlight_clusters=[0,2], highlight_min_width=5)`).

### `src/MMD.py`

- `MMDCalculator`: RBF MMD implementation with between/within bootstrap routines and comparison plots used in the notebooks.

### `src/regime_trading_pipeline.py`

- `RegimeRotationStrategy`: class encapsulating signal preparation, rolling WK-means fitting (with hot start), daily return construction, and backtesting for a growth vs. defensive rotation. `StrategyResult` includes strategy curve plus benchmark curves (`SPY`, `GrowthOnly`, `DefensiveOnly`).
- `_download_prices`: Yahoo Finance download helper with caching.
- `grid_search_regimes`: iterates across window/step/refit grids and reports metrics, with a `rich` progress bar.

## Notebooks (`jupyter/`)

- **`clustering_examples.ipynb`** – walkthrough of segmenting SPX returns, fitting Wasserstein vs. moment K-means, visualizing regimes, and comparing clustering quality via MMD.
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
