"""Regime-aware growth vs. defensive rotation strategy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import itertools

import numpy as np
import pandas as pd

from .clusterings import WassersteinKMeans
from .utils import segment_time_series

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


def _download_prices(tickers: Sequence[str], start: str, end: str, field: str) -> pd.DataFrame:
    if Path.exists(Path("../data/stocks.csv")):
        df = pd.read_csv("../data/stocks.csv", index_col=0, header=0).astype(float)
        df.index = pd.to_datetime(df.index)
        missing = [t for t in tickers if t not in df.columns]
        if not missing:
            return df

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


@dataclass
class StrategyResult:
    equity_curve: pd.Series
    benchmarks: Dict[str, pd.Series]
    signal_series: pd.Series
    weights: pd.DataFrame
    metrics: Dict[str, float]


class RegimeRotationStrategy:
    """Encapsulates WK-means training and portfolio construction."""

    def __init__(
        self,
        growth_tickers: Sequence[str],
        defensive_tickers: Sequence[str],
        start_date: str = "2014-05-25",  # 10 yrs data
        end_date: str = "2025-11-01",
        signal_csv: str | Path = "../data/SPX_hourly.csv",
        window: int = 36,
        step: int = 6,
        burn_in_segments: int = 700,  # approx. 2 yrs
        refit_every: int = 48,
        n_clusters: int = 3,
        p: int = 1,
        shift: bool = False,
    ) -> None:
        self.growth_tickers = list(growth_tickers)
        self.defensive_tickers = list(defensive_tickers)
        self.start_date = start_date
        self.end_date = end_date
        self.signal_csv = Path(signal_csv)
        self.window = window
        self.step = step
        self.burn_in_segments = burn_in_segments
        self.refit_every = refit_every
        self.n_clusters = n_clusters
        self.p = p
        self.shift = shift
        self.regime_series: Optional[pd.Series] = None
        self.growth_returns: Optional[pd.Series] = None
        self.defensive_returns: Optional[pd.Series] = None
        self.spy_returns: Optional[pd.Series] = None

    def _load_signal_returns(self) -> pd.Series:
        names = ["date", "time", "open", "high", "low", "close", "volume"]
        df = pd.read_csv(self.signal_csv, sep=";", names=names)
        df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"], format="%d/%m/%Y %H:%M", dayfirst=True)
        df = df.drop(columns=["date", "time"]).set_index("timestamp").sort_index()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        df = df[(self.end_date >= df.index) & (df.index >= self.start_date)]
        return df["close"].pct_change().dropna()

    def fit_wkmeans(self) -> pd.Series:
        signal_returns = self._load_signal_returns()
        segments = segment_time_series(signal_returns, self.window, self.step)
        if len(segments) <= self.burn_in_segments:
            raise ValueError("Not enough segments to cover burn-in period")

        labels: List[Optional[int]] = [None] * len(segments)
        idx = self.burn_in_segments
        prev_centroids: Optional[List[np.ndarray]] = None
        while idx < len(segments):
            history = segments.iloc[idx - self.burn_in_segments : idx].tolist()
            model = WassersteinKMeans(
                n_clusters=self.n_clusters,
                p=self.p,
                max_iter=500,
                tol=1e-6,
                random_state=42,
            )
            model.fit(history, initial_centroids=prev_centroids)
            prev_centroids = [c.copy() for c in model.centroids_]
            end = min(idx + self.refit_every, len(segments))
            preds = model.predict(segments.iloc[idx:end].tolist())
            for offset, lbl in enumerate(preds):
                labels[idx + offset] = int(lbl)
            idx = end

        regime_series = pd.Series(labels, index=segments.index, dtype="float").dropna().astype(int)
        regime_df = regime_series.to_frame(name="label")
        regime_df["effective_date"] = regime_df.index.normalize()
        # Roll signal to the next day if hour > 16
        regime_df.loc[regime_df.index.hour > 16, "effective_date"] += pd.offsets.BusinessDay(1)
        self.regime_series = regime_df.groupby("effective_date")["label"].last().sort_index()
        self._signal_returns = signal_returns
        return self.regime_series

    def build_returns(self) -> None:
        if self.regime_series is None:
            raise RuntimeError("Call fit_wkmeans() before build_returns().")
        start = self._signal_returns.index.min().strftime("%Y-%m-%d")
        end = self._signal_returns.index.max().strftime("%Y-%m-%d")
        tickers = sorted(set(self.growth_tickers) | set(self.defensive_tickers) | {"SPY"})
        close_prices = _download_prices(tickers, start=start, end=end, field="Close")
        returns = close_prices.pct_change().dropna()

        self.growth_returns = returns[list(self.growth_tickers)].mean(axis=1)
        self.defensive_returns = returns[list(self.defensive_tickers)].mean(axis=1)
        self.spy_returns = returns["SPY"]

    def backtest(
        self,
        allocations: Optional[Dict[int, Dict[str, float]]] = {
            0: {"growth": 1.0, "defensive": 0.0},
            1: {"growth": 0.5, "defensive": 0.5},
            2: {"growth": 0.0, "defensive": 1.0},
        },
    ) -> StrategyResult:
        if self.regime_series is None:
            raise RuntimeError("No regime labels. Call fit_wkmeans() first.")
        if self.growth_returns is None or self.defensive_returns is None or self.spy_returns is None:
            raise RuntimeError("Call build_returns() before backtest().")

        index = self.growth_returns.index.intersection(self.defensive_returns.index).intersection(self.spy_returns.index)
        regime_series = self.regime_series.reindex(index, method="ffill").dropna().astype(int)
        index = index.intersection(regime_series.index)
        regime_series = regime_series.loc[index]
        growth_returns = self.growth_returns.reindex(index).fillna(0.0)
        defensive_returns = self.defensive_returns.reindex(index).fillna(0.0)
        spy_returns = self.spy_returns.reindex(index).fillna(0.0)

        weights = pd.DataFrame(index=index, columns=["growth", "defensive"], data=0.0)
        for regime, alloc in allocations.items():
            mask = regime_series == regime
            for leg, val in alloc.items():
                weights.loc[mask, leg] = val
        if self.shift:
            weights = weights.shift(1).fillna(0)

        strategy_returns = weights["growth"] * growth_returns + weights["defensive"] * defensive_returns
        equity_curve = (1 + strategy_returns).cumprod()
        benchmark_curve = {
            "GrowthOnly": (1 + growth_returns).cumprod(),
            "EqualWeight": (1 + 0.5 * growth_returns + 0.5 * defensive_returns).cumprod(),
            "DefensiveOnly": (1 + defensive_returns).cumprod(),
            "SPY": (1 + spy_returns).cumprod(),
        }
        metrics = self._compute_metrics(strategy_returns)
        return StrategyResult(
            equity_curve=equity_curve,
            benchmarks=benchmark_curve,
            signal_series=regime_series,
            weights=weights,
            metrics=metrics,
        )

    @staticmethod
    def _compute_metrics(returns: pd.Series, periods_per_year: int = 252) -> Dict[str, float]:
        cumulative = (1 + returns).prod() - 1.0
        ann_return = (1 + cumulative) ** (periods_per_year / len(returns)) - 1.0 if len(returns) else 0.0
        ann_vol = returns.std(ddof=0) * np.sqrt(periods_per_year)
        sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
        curve = (1 + returns).cumprod()
        drawdown = curve / curve.cummax() - 1.0
        return {
            "start_date": returns.index[0].strftime("%Y-%m-%d"),
            "end_date": returns.index[-1].strftime("%Y-%m-%d"),
            "duration": (returns.index[-1] - returns.index[0]).days,
            "cumulative_return": cumulative,
            "annual_return": ann_return,
            "annual_volatility": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": drawdown.min(),
        }


def grid_search_regimes(
    growth,
    defensive,
    start_date="2014-05-15",
    p=2,
    windows=(360,),
    steps=(12,),
    refits=(96,),
):
    from rich.progress import Progress

    combos = list(itertools.product(windows, steps, refits))
    results = []
    index = []
    with Progress() as progress:
        task = progress.add_task("grid", total=len(combos))
        for win, step, refit in combos:
            strategy = RegimeRotationStrategy(
                growth_tickers=growth,
                defensive_tickers=defensive,
                start_date=start_date,
                window=win,
                step=step,
                refit_every=refit,
                p=p,
                shift=True,
            )
            try:
                strategy.fit_wkmeans()
                strategy.build_returns()
                result = strategy.backtest(
                    allocations={
                        0: {"growth": 1.0, "defensive": 0.0},
                        1: {"growth": 0.0, "defensive": 1.0},
                        2: {"growth": 0.0, "defensive": 1.0},
                    },
                )
            except Exception as exc:
                progress.console.print(f"[red]skip (win={win}, step={step}, refit={refit}): {exc}[/red]")
                progress.advance(task)
                continue

            results.append(result.metrics)
            index.append((win, step, refit))
            progress.advance(task)
            progress.refresh()

    idx = pd.MultiIndex.from_tuples(index, names=["window", "step", "refit_every"])
    return pd.DataFrame(results, index=idx).sort_values("sharpe", ascending=False)
