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
    score_series: pd.Series
    weights: pd.DataFrame
    metrics: Dict[str, float]
    allocations: Dict[int, Dict[str, float]]


class RegimeRotationStrategy:
    """Encapsulates WK-means / MK-means training and portfolio construction."""

    def __init__(
        self,
        growth_tickers: Sequence[str],
        defensive_tickers: Sequence[str],
        extra_legs: Optional[Dict[str, Sequence[str]]] = None,
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
        self.score_series: Optional[pd.Series] = None
        self._signal_returns: Optional[pd.Series] = None
        self.spy_returns: Optional[pd.Series] = None
        self.extra_legs = {name: list(ticks) for name, ticks in (extra_legs or {}).items()}
        self.leg_returns_modes: Dict[str, Dict[str, pd.Series]] = {}

    def fit_kmeans(self) -> pd.Series:
        signal_returns = load_signal(self.signal_csv, self.start_date, self.end_date)["Return"]
        segments = segment_time_series(signal_returns, self.window, self.step)
        if len(segments) <= self.burn_in_segments:
            raise ValueError("Not enough segments to cover burn-in period")

        labels: List[Optional[int]] = [None] * len(segments)
        scores: List[Optional[float]] = [None] * len(segments)
        idx = int(self.burn_in_segments - 1 * (self.refit_every == 0))
        prev_centroids: Optional[List[np.ndarray]] = None
        while idx < len(segments):
            history = segments.iloc[idx - self.burn_in_segments + 1 * (self.refit_every == 0) : idx + 1 * (self.refit_every == 0)].tolist()
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
            end = min(idx + max(self.refit_every, 1), len(segments))
            preds, sc = model.predict(segments.iloc[idx:end].tolist())
            for offset in range(len(preds)):
                labels[idx + offset] = int(preds[offset])
                scores[idx + offset] = sc[offset]
            idx = end

        regime_series = pd.Series(labels, index=segments.index, dtype="float").dropna().astype(int)
        scores_series = pd.Series(scores, index=segments.index, dtype="float").dropna().astype(float)
        regime_df = regime_series.to_frame(name="label")
        regime_df["score"] = scores_series
        regime_df["effective_date"] = regime_df.index.normalize()
        # Roll signal to the next day if hour > 16
        regime_df.loc[regime_df.index.hour > 16, "effective_date"] += pd.offsets.BusinessDay(1)
        daily_series = regime_df.groupby("effective_date")["label"].last().sort_index()
        if self.max_label_gap > 0:
            daily_series = smooth_labels(daily_series, self.max_label_gap)
        self.regime_series = daily_series
        self.score_series = regime_df.groupby("effective_date")["score"].last().sort_index()
        self._signal_returns = signal_returns
        return self.regime_series

    def build_returns(self) -> None:
        if self.regime_series is None:
            raise RuntimeError("Call fit_wkmeans() before build_returns().")
        start = self._signal_returns.index.min().strftime("%Y-%m-%d")
        end = self._signal_returns.index.max().strftime("%Y-%m-%d")
        all_leg_tickers = {
            "growth": self.growth_tickers,
            "defensive": self.defensive_tickers,
            **self.extra_legs,
        }
        ticker_set: set[str] = {"SPY"}
        for tickers in all_leg_tickers.values():
            ticker_set.update(tickers)
        tickers = sorted(ticker_set)
        close_prices = download_prices(tickers, start=start, end=end, field="Close")
        returns = close_prices.pct_change().fillna(0.0)

        market_caps = None
        if self.weighting == "market_cap":
            try:
                market_caps = download_market_caps(tickers, start=start, end=end)
            except Exception as exc:
                if self.weighting == "market_cap":
                    raise RuntimeError("Can not download market cap data!")

        if market_caps is not None and not market_caps.empty:
            market_caps = market_caps.reindex(returns.index).ffill()

        def _equal_weight(series_names: Sequence[str]) -> pd.Series:
            cols = [col for col in series_names if col in returns.columns]
            if not cols:
                raise ValueError("No overlapping tickers for returns calculation.")
            return returns[cols].mean(axis=1)

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

        leg_returns_modes: Dict[str, Dict[str, pd.Series]] = {}
        for leg_name, leg_tickers in all_leg_tickers.items():
            leg_modes = {"equal": _equal_weight(leg_tickers)}
            if market_caps is not None:
                mc = _mcap_weight(leg_tickers)
                if mc is not None:
                    leg_modes["market_cap"] = mc
            leg_returns_modes[leg_name] = leg_modes

        if "growth" not in leg_returns_modes or "defensive" not in leg_returns_modes:
            raise RuntimeError("Growth and defensive legs must have valid return series.")

        self.leg_returns_modes = leg_returns_modes

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
        if not self.leg_returns_modes or self.spy_returns is None:
            raise RuntimeError("Call build_returns() before backtest().")

        indices = [modes["equal"].index for modes in self.leg_returns_modes.values()] + [self.spy_returns.index]
        index = indices[0]
        for idx_series in indices[1:]:
            index = index.intersection(idx_series)
        regime_series = self.regime_series.reindex(index, method="ffill").dropna().astype(int)
        score_series = self.score_series.reindex(index, method="ffill").dropna().astype(float)
        index = index.intersection(regime_series.index)
        regime_series = regime_series.loc[index]
        score_series = score_series.loc[index]
        spy_returns = self.spy_returns.reindex(index).fillna(0.0)
        leg_returns: Dict[str, pd.Series] = {}
        for leg_name, modes in self.leg_returns_modes.items():
            series = modes.get(self.weighting)
            if series is None:
                continue
            leg_returns[leg_name] = series.reindex(index).fillna(0.0)
        if not leg_returns:
            raise RuntimeError("No leg return series available for backtest.")

        weights = pd.DataFrame(index=index, columns=leg_returns.keys(), data=0.0)
        for regime, alloc in allocations.items():
            mask = regime_series == regime
            for leg, val in alloc.items():
                if leg not in weights.columns:
                    raise KeyError(f"Allocation leg '{leg}' not available. Available legs: {list(weights.columns)}")
                weights.loc[mask, leg] = val

        if self.shift:
            weights = weights.shift(1).fillna(0)

        strategy_returns = sum(weights[col] * leg_returns[col] for col in weights.columns)
        benchmark_curve = {}
        if "growth" in leg_returns:
            benchmark_curve["GrowthOnly"] = (1 + leg_returns["growth"]).cumprod()
        if "growth" in leg_returns and "defensive" in leg_returns:
            benchmark_curve["EqualWeight"] = (1 + 0.5 * leg_returns["growth"] + 0.5 * leg_returns["defensive"]).cumprod()
        if "defensive" in leg_returns:
            benchmark_curve["DefensiveOnly"] = (1 + leg_returns["defensive"]).cumprod()
        benchmark_curve["SPY"] = (1 + spy_returns).cumprod()
        equity_curve = (1 + strategy_returns).cumprod()

        metrics = self._compute_metrics(strategy_returns)
        return StrategyResult(
            equity_curve=equity_curve,
            benchmarks=benchmark_curve,
            signal_series=regime_series,
            score_series=score_series,
            weights=weights,
            metrics=metrics,
            allocations=allocations
        )

    @staticmethod
    def _compute_metrics(returns: pd.Series, periods_per_year: int = 252) -> Dict[str, float]:
        cumulative = (1 + returns).prod() - 1.0
        ann_return = (1 + cumulative) ** (periods_per_year / len(returns)) - 1.0 if len(returns) else 0.0
        ann_vol = returns.std(ddof=1) * np.sqrt(periods_per_year)
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


__all__ = ["StrategyResult", "RegimeRotationStrategy"]


def main():
    import os

    os.chdir("./src")

    growth = [
        "IWP",   # iShares Russell Mid-Cap Growth ETF
        "IWY",   # iShares Russell Top 200 Growth ETF
        "QUAL",  # iShares MSCI USA Quality Factor ETF
        "QQQ",   # Invesco QQQ Trust (Nasdaq-100)
        "RPG",   # Invesco S&P 500 Pure Growth ETF
        "SCHG",  # Schwab U.S. Large-Cap Growth ETF
        "SCHM",  # Schwab U.S. Mid-Cap ETF
        "VBK",   # Vanguard Small-Cap Growth ETF
        "VGT",   # Vanguard Information Technology Index Fund ETF
        "VUG",   # Vanguard Growth ETF
    ]
    # Retrieved from https://www.simplysafedividends.com/world-of-dividends/posts/939-20-best-recession-proof-dividend-stocks-for-a-2025-downturn
    defensive = [
        "BMY",  # Bristol-Myers Squibb Co. (pharmaceuticals)
        "CL",   # Colgate-Palmolive Co. (consumer staples – personal care)
        "CVX",  # Chevron Corp. (integrated oil & gas)
        "DUK",  # Duke Energy Corp. (regulated electric utility)
        "ED",   # Consolidated Edison, Inc. (regulated utility)
        "EPD",  # Enterprise Products Partners L.P. (midstream energy MLP)
        "GIS",  # General Mills, Inc. (packaged foods)
        "GPC",  # Genuine Parts Co. (industrial/auto parts distributor)
        "JNJ",  # Johnson & Johnson (healthcare & consumer health)
        "KMB",  # Kimberly-Clark Corp. (tissue & hygiene products)
        "KO",   # Coca-Cola Co. (beverages)
        "LMT",  # Lockheed Martin Corp. (defense & aerospace)
        "MCD",  # McDonald's Corp. (global quick-service restaurants)
        "O",    # Realty Income Corp. (net-lease REIT, “monthly dividend”)
        "PEP",  # PepsiCo, Inc. (snacks & beverages)
        "PG",   # Procter & Gamble Co. (household & personal products)
        "PSA",  # Public Storage (self-storage REIT)
        "VZ",   # Verizon Communications Inc. (telecom)
        "WEC",  # WEC Energy Group, Inc. (regulated utility)
        "XOM",  # Exxon Mobil Corp. (integrated oil & gas)
    ]

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
