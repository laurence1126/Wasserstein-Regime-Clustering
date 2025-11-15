# `src/regime_trading_pipeline.py`

Implements a Wasserstein K-means driven rotation between growth and defensive equity baskets.

## `StrategyResult`
Simple dataclass with:
- `equity_curve` (pd.Series)
- `benchmarks`: dictionary of benchmark curves (e.g., SPY, GrowthOnly, DefensiveOnly)
- `signal_series` (daily regimes)
- `weights` (growth/defensive exposures)
- `metrics` dictionary (cumulative, annualized stats, Sharpe, drawdown, sample dates)

## `RegimeRotationStrategy`
Encapsulates the workflow.

### Constructor arguments
- `growth_tickers`, `defensive_tickers`: lists of tradable symbols.
- `start_date`: earliest timestamp for the signal CSV.
- `signal_csv`: path to SPX hourly data.
- `window`, `step`: sliding-window parameters for segmentation.
- `burn_in_segments`: minimum history used before generating labels (rolling training window).
- `refit_every`: number of segments predicted per refit.
- `n_clusters`, `p`: WK-means configuration.
- `shift`: optional 1-day lag of weights.
- `weighting`: `"equal"` for basket means, `"market_cap"` to weight constituents by their market capitalization (from `download_market_caps`).

- `_load_signal_returns()`: parses the CSV and returns hourly log returns filtered by `start_date`.
- `fit_wkmeans()`: calls the Series-based `segment_time_series`, runs rolling WK-means with hot-started centroids, and builds a regime Series indexed by the segment end timestamps (labels shift to the next day when timestamp > 16:00).
- `build_returns()`: downloads daily closes (via `download_prices`) and computes equal-weighted growth/defensive returns along with SPY; if market-cap data are available the method also stores cap-weighted legs and selects whichever scheme matches `weighting`.
- `backtest(allocations=None)`: forward-fills regimes over the return index, builds weight series from the allocation map (optional 1-day shift), and produces the strategy curve plus multiple benchmarks (SPY plus Growth/Defensive/Equal baskets for every available weighting scheme).
- `_compute_metrics()`: helper returning date span, cumulative/annual performance, volatility, Sharpe, and max drawdown.

## `grid_search_regimes`
Convenience routine to sweep combinations of window/step/refit parameters. Displays a Rich progress bar, handles exceptions gracefully, and returns a MultiIndex DataFrame sorted by Sharpe.

## Usage example
```python
from src.regime_trading_pipeline import RegimeRotationStrategy

strategy = RegimeRotationStrategy(
    growth_tickers=growth_list,
    defensive_tickers=defensive_list,
    start_date="2014-05-15",
    window=72,
    step=12,
    burn_in_segments=500,
    refit_every=48,
    n_clusters=3,
    p=2,
)
strategy.fit_wkmeans()
strategy.build_returns()
result = strategy.backtest()
```

For systematic tuning, call `grid_search_regimes(...)` with desired parameter grids.
