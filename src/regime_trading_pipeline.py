"""Regime-aware growth vs. defensive rotation strategy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Literal
import itertools

import numpy as np
import pandas as pd

from .clustering_methods import WassersteinKMeans, MomentKMeans
from .utils import segment_time_series, download_prices, download_market_caps, load_signal, smooth_labels


@dataclass
class StrategyResult:
    equity_curve: pd.Series
    benchmarks: Dict[str, pd.Series]
    signal_series: pd.Series
    weights: pd.DataFrame
    metrics: Dict[str, float]
    allocations: Dict[int, Dict[str, float]]


class RegimeRotationStrategy:
    """Encapsulates WK-means / MK-means training and portfolio construction."""

    def __init__(
        self,
        growth_tickers: Sequence[str],
        defensive_tickers: Sequence[str],
        start_date: str = "2014-05-25",  # 10 yrs data
        end_date: str = "2025-11-01",
        signal_csv: str | Path = "../data/SPX_hourly.csv",
        window: int = 24,
        step: int = 6,
        burn_in_segments: int = 700,  # approx. 2 yrs
        refit_every: int = 48,
        n_clusters: int = 3,
        p_dim: int = 2,
        shift: bool = False,
        max_label_gap: int = 0,
        weighting: Literal["equal", "market_cap"] = "equal",
        distance: Literal["wasserstein", "moment"] = "wasserstein",
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
        self.p_dim = p_dim
        self.shift = shift
        self.max_label_gap = int(max_label_gap)
        self.weighting = weighting.lower()
        if self.weighting not in {"equal", "market_cap"}:
            raise ValueError("weighting must be 'equal' or 'market_cap'")
        self.distance = distance.lower()
        if self.distance not in {"wasserstein", "moment"}:
            raise ValueError("distance must be 'wasserstein' or 'moment'")
        self.regime_series: Optional[pd.Series] = None
        self._signal_returns: Optional[pd.Series] = None
        self.growth_returns_modes: Dict[str, pd.Series] = {}
        self.defensive_returns_modes: Dict[str, pd.Series] = {}
        self.spy_returns: Optional[pd.Series] = None

    def fit_kmeans(self) -> pd.Series:
        signal_returns = load_signal(self.signal_csv, self.start_date, self.end_date)["Return"]
        segments = segment_time_series(signal_returns, self.window, self.step)
        if len(segments) <= self.burn_in_segments:
            raise ValueError("Not enough segments to cover burn-in period")

        labels: List[Optional[int]] = [None] * len(segments)
        idx = self.burn_in_segments
        prev_centroids: Optional[List[np.ndarray]] = None
        while idx < len(segments):
            history = segments.iloc[idx - self.burn_in_segments : idx].tolist()
            if self.distance == "wasserstein":
                model = WassersteinKMeans(
                    n_clusters=self.n_clusters,
                    p_dim=self.p_dim,
                    max_iter=500,
                    random_state=42,
                )
                model.fit(history, initial_centroids=prev_centroids)
                prev_centroids = [c.copy() for c in model.centroids_]
            elif self.distance == "moment":
                model = MomentKMeans(
                    n_clusters=self.n_clusters,
                    p_dim=self.p_dim,
                    max_iter=500,
                    random_state=42,
                )
                model.fit(history)
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
        daily_series = regime_df.groupby("effective_date")["label"].last().sort_index()
        if self.max_label_gap > 0:
            daily_series = smooth_labels(daily_series, self.max_label_gap)
        self.regime_series = daily_series
        self._signal_returns = signal_returns
        return self.regime_series

    def build_returns(self) -> None:
        if self.regime_series is None:
            raise RuntimeError("Call fit_wkmeans() before build_returns().")
        start = self._signal_returns.index.min().strftime("%Y-%m-%d")
        end = self._signal_returns.index.max().strftime("%Y-%m-%d")
        tickers = sorted(set(self.growth_tickers) | set(self.defensive_tickers) | {"SPY"})
        close_prices = download_prices(tickers, start=start, end=end, field="Close")
        returns = close_prices.pct_change().fillna(0.0)

        def _equal_weight(series_names: Sequence[str]) -> pd.Series:
            cols = [col for col in series_names if col in returns.columns]
            if not cols:
                raise ValueError("No overlapping tickers for returns calculation.")
            return returns[cols].mean(axis=1)

        market_caps = None
        try:
            market_caps = download_market_caps(tickers, start=start, end=end)
        except Exception as exc:
            if self.weighting == "market_cap":
                raise RuntimeError("Can not download market cap data!")

        if market_caps is not None and not market_caps.empty:
            market_caps = market_caps.reindex(returns.index).ffill()

        def _mcap_weight(series_names: Sequence[str]) -> Optional[pd.Series]:
            if market_caps is None or market_caps.empty:
                return None
            cols = [col for col in series_names if col in returns.columns and col in market_caps.columns]
            if not cols:
                return None
            caps = market_caps[cols]
            weights = caps.div(caps.sum(axis=1), axis=0).replace([np.inf, -np.inf], np.nan).shift(1).fillna(0.0)
            weighted_returns = (returns[cols] * weights).sum(axis=1)
            return weighted_returns

        growth_equal = _equal_weight(self.growth_tickers)
        defensive_equal = _equal_weight(self.defensive_tickers)
        growth_mcap = _mcap_weight(self.growth_tickers)
        defensive_mcap = _mcap_weight(self.defensive_tickers)

        self.growth_returns_modes = {"equal": growth_equal}
        self.defensive_returns_modes = {"equal": defensive_equal}
        if growth_mcap is not None:
            self.growth_returns_modes["market_cap"] = growth_mcap
        if defensive_mcap is not None:
            self.defensive_returns_modes["market_cap"] = defensive_mcap

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
        if self.growth_returns_modes is None or self.defensive_returns_modes is None or self.spy_returns is None:
            raise RuntimeError("Call build_returns() before backtest().")

        index = (
            self.growth_returns_modes["equal"].index.intersection(self.defensive_returns_modes["equal"].index).intersection(self.spy_returns.index)
        )
        regime_series = self.regime_series.reindex(index, method="ffill").dropna().astype(int)
        index = index.intersection(regime_series.index)
        regime_series = regime_series.loc[index]
        spy_returns = self.spy_returns.reindex(index).fillna(0.0)
        growth_equal = self.growth_returns_modes["equal"].reindex(index).fillna(0.0)
        defensive_equal = self.defensive_returns_modes["equal"].reindex(index).fillna(0.0)
        growth_mcap = self.growth_returns_modes.get("market_cap")
        defensive_mcap = self.defensive_returns_modes.get("market_cap")
        if growth_mcap is not None:
            growth_mcap = growth_mcap.reindex(index).fillna(0.0)
        if defensive_mcap is not None:
            defensive_mcap = defensive_mcap.reindex(index).fillna(0.0)

        weights = pd.DataFrame(index=index, columns=["growth", "defensive"], data=0.0)
        for regime, alloc in allocations.items():
            mask = regime_series == regime
            for leg, val in alloc.items():
                weights.loc[mask, leg] = val

        if self.shift:
            weights = weights.shift(1).fillna(0)

        if self.weighting == "equal":
            strategy_returns = weights["growth"] * growth_equal + weights["defensive"] * defensive_equal
            benchmark_curve = {
                "GrowthOnly": (1 + growth_equal).cumprod(),
                "EqualWeight": (1 + 0.5 * growth_equal + 0.5 * defensive_equal).cumprod(),
                "DefensiveOnly": (1 + defensive_equal).cumprod(),
                "SPY": (1 + spy_returns).cumprod(),
            }
        elif self.weighting == "market_cap":
            strategy_returns = weights["growth"] * growth_mcap + weights["defensive"] * defensive_mcap
            benchmark_curve = {
                "GrowthOnly": (1 + growth_mcap).cumprod(),
                "EqualWeight": (1 + 0.5 * growth_mcap + 0.5 * defensive_mcap).cumprod(),
                "DefensiveOnly": (1 + defensive_mcap).cumprod(),
                "SPY": (1 + spy_returns).cumprod(),
            }
        equity_curve = (1 + strategy_returns).cumprod()

        metrics = self._compute_metrics(strategy_returns)
        return StrategyResult(
            equity_curve=equity_curve,
            benchmarks=benchmark_curve,
            signal_series=regime_series,
            weights=weights,
            metrics=metrics,
            allocations=allocations
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

    @staticmethod
    def grid_search_regimes(
        growth_tickers: Sequence[str],
        defensive_tickers: Sequence[str],
        start_date: str = "2014-05-25",
        end_date: str = "2025-11-01",
        p_dims=(2,),
        windows=(360,),
        steps=(12,),
        refits=(96,),
        burn_ins=(700,),
    ):
        from rich.progress import Progress

        combos = list(itertools.product(p_dims, windows, steps, refits, burn_ins))
        results = []
        index = []
        with Progress() as progress:
            task = progress.add_task("grid", total=len(combos))
            for p, win, step, refit, burn_in in combos:
                strategy = RegimeRotationStrategy(
                    growth_tickers=growth_tickers,
                    defensive_tickers=defensive_tickers,
                    start_date=start_date,
                    end_date=end_date,
                    window=win,
                    step=step,
                    refit_every=refit,
                    p_dim=p,
                    burn_in_segments=burn_in,
                    shift=True
                )
                try:
                    strategy.fit_kmeans()
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
                index.append((p, win, step, refit, burn_in))
                progress.advance(task)
                progress.refresh()

        idx = pd.MultiIndex.from_tuples(index, names=["p", "window", "step", "refit_every", "burn_in"])
        return pd.DataFrame(results, index=idx).sort_values("sharpe", ascending=False)


def main():
    import os

    os.chdir("./src")

    growth = ["ADBE", "CRM", "LULU", "ORLY", "COST", "TMO", "LIN", "ACN", "MA", "V", "SPGI", "MCO", "DHR", "SHW", "INTU", "NFLX", "NOW", "SNPS", "ISRG", "CDNS"]
    defensive = ["EPD", "VZ", "O", "GIS", "BMY", "KMB", "CVX", "PSA", "PEP", "XOM", "DUK", "ED", "GPC", "WEC", "LMT", "KO", "PG", "JNJ", "CL", "MCD"]

    strategy = RegimeRotationStrategy(
        growth_tickers=growth,
        defensive_tickers=defensive,
        start_date="2019-05-09",  # 5 yrs data (post covid)
        p_dim=2,                  # W2 distance
        window=360,               # approx. 15 days hourly return
        step=12,                  # half a day
        refit_every=48,           # refit MK-means every 24 days
        shift=True                # avoid using future information
    )

    strategy.fit_kmeans()
    strategy.build_returns()

    result = strategy.backtest(
        allocations={
            0: {"growth": 1.0, "defensive": 0.0},
            1: {"growth": 0.0, "defensive": 1.0},
            2: {"growth": 0.0, "defensive": 1.0},
        },
    )

    for k, v in result.metrics.items():
        if isinstance(v, str) or isinstance(v, int):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
