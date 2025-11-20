"""Performance-analysis and visualization helpers for RegimeRotationStrategy results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import CLUSTER_PALETTE
from .regime_trading_pipeline import StrategyResult


def _prep_returns(curve: pd.Series) -> pd.Series:
    """Convert an equity curve into simple returns."""
    if curve is None or curve.empty:
        return pd.Series(dtype=float)
    returns = curve.sort_index().pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    return returns


@dataclass
class RegimePerformanceToolkit:
    """Convenience wrapper around StrategyResult with rich analytics/plots."""

    result: StrategyResult
    periods_per_year: int = 252

    def __post_init__(self) -> None:
        self.equity_curve = self.result.equity_curve.sort_index()
        self.strategy_returns = _prep_returns(self.equity_curve)
        self.benchmark_curves = {name: series.sort_index() for name, series in self.result.benchmarks.items()}
        self.benchmark_returns = {name: _prep_returns(series) for name, series in self.benchmark_curves.items()}
        self.weights = self.result.weights.reindex(self.equity_curve.index).ffill().fillna(0.0)
        signal = self.result.signal_series.reindex(self.equity_curve.index, method="ffill")
        scores = self.result.score_series.reindex(self.equity_curve.index, method="ffill")
        self.signal_series = signal.astype(int) if signal.notna().any() else signal
        self.score_series = scores.astype(float) if scores.notna().any() else scores

    # ------------------------------------------------------------------
    # Metric tables
    # ------------------------------------------------------------------
    def show_basic_info(self):
        print("Start date:", self.result.metrics["start_date"])
        print("End date:  ", self.result.metrics["end_date"])
        print("Duration:  ", (pd.to_datetime(self.result.metrics["end_date"]) - pd.to_datetime(self.result.metrics["start_date"])).days, "\n")
        for i, cnt in enumerate(np.bincount(self.result.signal_series)):
            print("# of points in cluster", i, ":", cnt)

    def summary_table(self, include_benchmarks: bool = True) -> pd.DataFrame:
        """Return extended stats for the strategy (plus optional benchmarks)."""

        rows = []
        rows.append(("Strategy", self._extended_metrics(self.strategy_returns, self.equity_curve)))
        if include_benchmarks:
            for name, returns in self.benchmark_returns.items():
                rows.append((name, self._extended_metrics(returns, self.benchmark_curves[name])))
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame({name: stats for name, stats in rows}).T
        percent_cols = [
            "cumulative_return",
            "annual_return",
            "annual_vol",
            "max_drawdown",
            "hit_rate",
            "avg_up_day",
            "avg_down_day",
        ]
        return self._format_percent_columns(df, percent_cols)

    def monthly_returns_table(self) -> pd.DataFrame:
        """Heat-map style table of monthly returns (rows=years, cols=months)."""
        if self.strategy_returns.empty:
            return pd.DataFrame()
        monthly = self.strategy_returns.resample("ME").apply(lambda s: (1 + s).prod() - 1)
        df = monthly.to_frame(name="return")
        df["year"] = df.index.year
        df["month"] = df.index.strftime("%b")
        pivot = df.pivot(index="year", columns="month", values="return")
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        pivot = pivot.reindex(columns=months)
        ytd = df.groupby("year")["return"].apply(lambda s: (1 + s).prod() - 1)
        pivot["YTD"] = ytd
        pivot = pivot.sort_index()
        return pivot.map(self._format_percent_value)

    def drawdown_table(self, top_n: int = 5) -> pd.DataFrame:
        """Tabular view of the largest drawdowns."""
        records = self._drawdown_periods(self.equity_curve)
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records).sort_values("depth").head(top_n)
        df["depth"] = df["depth"].apply(self._format_percent_value)
        return df.set_index("rank")

    def regime_transition_matrix(self, normalize: bool = True) -> pd.DataFrame:
        """Return a regime-to-regime transition matrix."""
        series = self.signal_series
        if series is None or series.dropna().empty:
            return pd.DataFrame()
        prev = series.shift(1).dropna().astype(int)
        curr = series.loc[prev.index].astype(int)
        matrix = pd.crosstab(prev, curr).sort_index(axis=0).sort_index(axis=1)
        if normalize and not matrix.empty:
            matrix = matrix.div(matrix.sum(axis=1), axis=0)
        matrix.index.name = "from"
        matrix.columns.name = "to"
        return matrix

    def average_holding_period(self, tol: float = 1e-6) -> pd.DataFrame:
        """Holding-period stats grouped by unique allocation vectors."""
        weights = self.weights.fillna(0.0)[1:]
        if weights.empty:
            return pd.DataFrame(columns=["avg", "min", "max", "count"])

        records: List[tuple[tuple[float, ...], int]] = []
        prev_key: Optional[tuple[float, ...]] = None
        run = 0
        for _, row in weights.iterrows():
            key = tuple(np.round(row.values, 6))
            if prev_key is None:
                prev_key = key
                run = 1
                continue
            if all(abs(a - b) <= tol for a, b in zip(prev_key, key)):
                run += 1
            else:
                records.append((prev_key, run))
                prev_key = key
                run = 1
        if prev_key is not None and run:
            records.append((prev_key, run))

        lengths: Dict[tuple[float, ...], List[int]] = {}
        for key, length in records:
            lengths.setdefault(key, []).append(length)

        if not lengths:
            return pd.DataFrame(columns=["avg", "min", "max", "count"])

        match_key = {col: col[:1].upper() for col in weights.columns}
        rows = []
        for key, runs in lengths.items():
            arr = np.array(runs, dtype=float)
            rows.append(
                {
                    "holding_periods": ", ".join(f"{match_key[col]}={val:.1f}" for col, val in zip(weights.columns, key)),
                    "avg": round(arr.mean(), 2),
                    "min": int(arr.min()),
                    "max": int(arr.max()),
                    "count": int(len(arr)),
                }
            )
        return pd.DataFrame(rows).set_index("holding_periods")

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------
    def plot_equity_curves(self, type: str) -> plt.Figure:
        drawdown = self._drawdown_series(self.equity_curve)
        min_idx = drawdown.idxmin()
        max_idx = self.equity_curve.loc[:min_idx].idxmax()

        fig, (ax_equity, ax_weight) = plt.subplots(2, 1, sharex=True, figsize=(12, 6), gridspec_kw={"height_ratios": (3, 1)})

        # top panel: equity curves
        ax_equity.plot(self.equity_curve.index, self.equity_curve, label="Strategy", zorder=100, color=CLUSTER_PALETTE[0])
        ax_equity.axvspan(max_idx, min_idx, color="red", alpha=0.15, label="Max DD")
        cols = ["gray", "darkgray", "lightgray", CLUSTER_PALETTE[2]]
        for i, (k, v) in enumerate(self.result.benchmarks.items()):
            ax_equity.plot(v.index, v, label=k, color=cols[i], zorder=-i)
        ax_equity.set_title(f"Equity Curve ({type})")
        ax_equity.grid(alpha=0.6)
        ax_equity.legend()

        # bottom panel: signal/weight series
        ax_weight.plot(self.signal_series.index, self.signal_series, color=CLUSTER_PALETTE[1], lw=1)
        ax_score = ax_weight.twinx()
        ax_score.plot(self.score_series.index, self.score_series, color=CLUSTER_PALETTE[4], lw=1, alpha=0.8)
        miny, _ = ax_score.get_ylim()
        ax_score.fill_between(
            self.score_series.index,
            self.score_series,
            miny,
            color=CLUSTER_PALETTE[4],
            alpha=0.15,
        )
        ax_weight.set_zorder(ax_score.get_zorder() + 1)
        ax_weight.patch.set_alpha(0)
        ax_score.set_ylabel("Distance Score")
        ax_weight.set_ylabel("Regimes")
        ax_weight.set_xlabel("Date")
        ax_weight.grid(alpha=0.6)

        plt.tight_layout()
        return fig

    def plot_rolling_metrics(self, window: int = 63) -> plt.Figure:
        """Plot rolling Sharpe and volatility series."""
        if self.strategy_returns.empty:
            raise ValueError("No returns to plot.")
        rolling_mean = self.strategy_returns.rolling(window).mean()
        rolling_std = self.strategy_returns.rolling(window).std(ddof=1)
        rolling_sharpe = rolling_mean / rolling_std * np.sqrt(self.periods_per_year)
        rolling_vol = rolling_std * np.sqrt(self.periods_per_year)

        fig, (ax_sharpe, ax_vol) = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
        ax_sharpe.plot(rolling_sharpe.index, rolling_sharpe, color=CLUSTER_PALETTE.get(4, "C4"))
        ax_sharpe.axhline(0, color="black", lw=0.8, ls="--")
        ax_sharpe.set_title(f"{window}-Day Rolling Sharpe")
        ax_sharpe.grid(alpha=0.4)

        ax_vol.plot(rolling_vol.index, rolling_vol, color=CLUSTER_PALETTE.get(3, "C3"))
        ax_vol.set_title(f"{window}-Day Rolling Volatility (annualized)")
        ax_vol.set_xlabel("Date")
        ax_vol.grid(alpha=0.4)

        plt.tight_layout()
        return fig

    def plot_weight_stack(self) -> plt.Figure:
        """Stack plot of portfolio weights through time."""
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.stackplot(
            self.weights.index,
            [self.weights[col] for col in self.weights.columns],
            labels=self.weights.columns,
            colors=[CLUSTER_PALETTE.get(idx, None) for idx, _ in enumerate(self.weights.columns)],
            alpha=0.8,
        )
        ax.set_ylabel("Weight")
        ax.set_title("Strategy Allocations")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper right", frameon=False)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_correlation_and_transitions(self) -> plt.Figure:
        """Side-by-side return correlations and regime transition probabilities."""
        frames = [self.strategy_returns.rename("Strategy")]
        for name, returns in self.benchmark_returns.items():
            frames.append(returns.rename(name))
        data = pd.concat(frames, axis=1).dropna()
        if data.empty:
            raise ValueError("Not enough overlapping returns for correlation plot.")
        corr = data.corr()

        matrix = self.regime_transition_matrix(normalize=True)
        if matrix.empty:
            raise ValueError("Transition matrix is empty.")

        fig, (ax_corr, ax_trans) = plt.subplots(1, 2, figsize=(12, 6))

        cax_corr = ax_corr.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax_corr.set_xticks(range(len(corr.columns)))
        ax_corr.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax_corr.set_yticks(range(len(corr.index)))
        ax_corr.set_yticklabels(corr.index)
        ax_corr.set_title("Return Correlations")
        fig.colorbar(cax_corr, ax=ax_corr, shrink=0.8)

        cax_trans = ax_trans.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        ticks = range(len(matrix.index))
        ax_trans.set_xticks(ticks)
        ax_trans.set_xticklabels(matrix.columns)
        ax_trans.set_yticks(ticks)
        ax_trans.set_yticklabels(matrix.index)
        ax_trans.set_xlabel("Next Regime")
        ax_trans.set_ylabel("Current Regime")
        ax_trans.set_title("Regime Transition Probabilities")
        fig.colorbar(cax_trans, ax=ax_trans, shrink=0.75)
        for (i, j), val in np.ndenumerate(matrix.values):
            ax_trans.text(j, i, f"{val:.4f}", ha="center", va="center", color="black")

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _extended_metrics(self, returns: pd.Series, curve: pd.Series) -> Dict[str, float]:
        if returns.empty:
            return {}
        cumulative = (1 + returns).prod() - 1.0
        ann_return = (1 + cumulative) ** (self.periods_per_year / len(returns)) - 1.0
        ann_vol = returns.std(ddof=1) * np.sqrt(self.periods_per_year)
        sharpe = ann_return / ann_vol if ann_vol and ann_vol > 0 else np.nan

        downside = returns[returns < 0]
        downside_vol = downside.std(ddof=1) * np.sqrt(self.periods_per_year) if not downside.empty else np.nan
        sortino = ann_return / downside_vol if downside_vol and downside_vol > 0 else np.nan

        curve = curve.reindex(returns.index, method="ffill")
        drawdown = self._drawdown_series(curve)
        max_dd = drawdown.min()

        hit_rate = (returns > 0).mean()
        avg_up = returns[returns > 0].mean()
        avg_down = returns[returns < 0].mean()

        metrics = {
            "cumulative_return": cumulative,
            "annual_return": ann_return,
            "annual_vol": ann_vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "hit_rate": hit_rate,
            "avg_up_day": avg_up,
            "avg_down_day": avg_down,
            "skew": returns.skew(),
            "kurtosis": returns.kurtosis(),
        }

        for k, v in metrics.items():
            metrics[k] = round(v, 6)

        return metrics

    def _align_returns_with_regimes(self) -> pd.DataFrame:
        if self.strategy_returns.empty or self.signal_series is None:
            return pd.DataFrame()
        aligned = pd.DataFrame({"returns": self.strategy_returns})
        aligned["regime"] = self.signal_series.reindex(aligned.index, method="ffill")
        aligned = aligned.dropna(subset=["regime"])
        if aligned.empty:
            return aligned
        aligned["regime"] = aligned["regime"].astype(int)
        for col in self.weights.columns:
            aligned[col] = self.weights[col].reindex(aligned.index, method="ffill")
        return aligned.dropna()

    @staticmethod
    def _drawdown_series(curve: pd.Series) -> pd.Series:
        if curve is None or curve.empty:
            return pd.Series(dtype=float)
        running_max = curve.cummax()
        drawdown = curve / running_max - 1.0
        return drawdown.fillna(0.0)

    @staticmethod
    def _drawdown_periods(curve: pd.Series) -> List[Dict[str, float]]:
        drawdown = RegimePerformanceToolkit._drawdown_series(curve)
        records = []
        in_dd = False
        start = trough = recover = None
        min_dd = 0.0
        rank = 1
        for timestamp, dd in drawdown.items():
            if dd < 0 and not in_dd:
                in_dd = True
                start = timestamp
                trough = timestamp
                min_dd = dd
            elif in_dd and dd < min_dd:
                min_dd = dd
                trough = timestamp
            elif in_dd and dd == 0:
                recover = timestamp
                records.append(
                    {
                        "rank": rank,
                        "start": start,
                        "trough": trough,
                        "recovery": recover,
                        "depth": min_dd,
                        "time_to_trough": (trough - start).days if trough and start else np.nan,
                        "recovery_days": (recover - trough).days if recover and trough else np.nan,
                        "duration": (recover - start).days if recover and start else np.nan,
                    }
                )
                rank += 1
                in_dd = False
                start = trough = recover = None
        if in_dd and start is not None:
            records.append(
                {
                    "rank": rank,
                    "start": start,
                    "trough": trough,
                    "recovery": pd.NaT,
                    "depth": min_dd,
                    "time_to_trough": (trough - start).days if trough and start else np.nan,
                    "recovery_days": np.nan,
                    "duration": np.nan,
                }
            )
        return records

    @staticmethod
    def _format_percent_value(value: float, decimals: int = 4) -> str:
        if value is None or (isinstance(value, str) and not value):
            return "NaN"
        if pd.isna(value):
            return "NaN"
        return f"{value * 100:.{decimals}f}%"

    def _format_percent_columns(self, df: pd.DataFrame, columns: Iterable[str], decimals: int = 4) -> pd.DataFrame:
        formatted = df.copy()
        for col in columns:
            if col in formatted.columns:
                formatted[col] = formatted[col].apply(lambda x: self._format_percent_value(x, decimals))
        return formatted


__all__ = ["RegimePerformanceToolkit"]
