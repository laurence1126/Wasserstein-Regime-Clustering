"""Merton jump-diffusion simulators, benchmarks, and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from .clustering_methods import HMMClustering, MomentKMeans, WassersteinKMeans


class JumpDiffusionParams(NamedTuple):
    """Container for (μ, σ, λ, γ, δ) defining a Merton jump diffusion."""

    mu: float
    sigma: float
    lam: float
    gamma: float
    delta: float


def _coerce_params(theta: Union[JumpDiffusionParams, Sequence[float]]) -> JumpDiffusionParams:
    if isinstance(theta, JumpDiffusionParams):
        return theta
    if isinstance(theta, (list, tuple, np.ndarray)):
        if len(theta) != 5:
            raise ValueError("Jump diffusion parameters must have five entries (mu, sigma, lambda, gamma, delta).")
        return JumpDiffusionParams(*map(float, theta))
    raise TypeError("Jump diffusion parameters must be a JumpDiffusionParams or a length-5 tuple/list/array.")


@dataclass(frozen=True)
class MertonJumpDiffusion:
    """Classic single-regime Merton jump diffusion."""

    params: Union[JumpDiffusionParams, Sequence[float]]

    def __post_init__(self):
        object.__setattr__(self, "params", _coerce_params(self.params))

    def log_return_moments(self, dt: float) -> Tuple[float, float]:
        """Analytical mean/variance of log-returns over step size dt."""
        if dt <= 0:
            raise ValueError("dt must be positive.")
        mu, sigma, lam, gamma, delta = self.params
        mean = ((mu - 0.5 * sigma**2) + lam * gamma) * dt
        variance = (sigma**2 + lam * (delta**2 + gamma**2)) * dt
        return mean, variance

    def simulate_path(self, T: float = 1.0, N: int = 252, S0: float = 100.0, random_state: Optional[int] = None):
        """Simulate a price path under a fixed-parameter Merton jump diffusion."""
        if T <= 0:
            raise ValueError("T must be positive.")
        if N <= 0:
            raise ValueError("N must be positive.")
        if S0 <= 0:
            raise ValueError("S0 must be positive.")

        rng = np.random.default_rng(random_state)
        dt = T / N
        times = np.linspace(0.0, T, N + 1)
        prices = np.empty(N + 1)
        prices[0] = S0
        log_price = np.log(S0)
        jumps = 0

        mu, sigma, lam, gamma, delta = self.params
        drift = (mu - 0.5 * sigma**2) * dt
        vol_term = sigma * np.sqrt(dt)

        for t in range(1, N + 1):
            diffusion = drift + vol_term * rng.standard_normal()
            n_jumps = rng.poisson(lam * dt)
            jump_term = rng.normal(gamma, delta, n_jumps).sum() if n_jumps else 0.0
            jumps += n_jumps
            log_price += diffusion + jump_term
            prices[t] = np.exp(log_price)

        return times, prices, jumps


class RegimeSwitchingMerton:
    """
    Regime-switching simulator that alternates between two Merton jump diffusions.
    Used to mimic Section 3.3.2 of the paper (bull/bear regimes with random change points).
    """

    def __init__(
        self,
        theta_bull: Union[JumpDiffusionParams, Sequence[float]],
        theta_bear: Union[JumpDiffusionParams, Sequence[float]],
        n_regime_switches: int = 10,
        regime_length_years: float = 0.5,
        length_jitter: float = 0.5,
        jitter_fraction: float = 0.5,
    ):
        if regime_length_years <= 0:
            raise ValueError("regime_length_years must be positive.")
        if n_regime_switches < 0:
            raise ValueError("n_regime_switches must be >= 0.")
        if not (0.0 <= jitter_fraction <= 1.0):
            raise ValueError("jitter_fraction must lie in [0, 1].")
        if length_jitter < 0:
            raise ValueError("length_jitter must be non-negative.")

        self.theta_bull = _coerce_params(theta_bull)
        self.theta_bear = _coerce_params(theta_bear)
        self.n_regime_switches = int(n_regime_switches)
        self.regime_length_years = float(regime_length_years)
        self.length_jitter = float(length_jitter)
        self.jitter_fraction = float(jitter_fraction)

    def _build_mask(self, total_steps: int, regime_steps: int, rng: np.random.Generator):
        mask = np.zeros(total_steps, dtype=int)
        intervals: List[Tuple[int, int]] = []
        if self.n_regime_switches == 0:
            return mask, intervals

        max_length = regime_steps
        if self.length_jitter > 0:
            max_length = int(round(regime_steps * (1 + self.length_jitter)))
            max_length = max(max_length, 1)
        available = total_steps - self.n_regime_switches * max_length
        if available < 0:
            raise ValueError("Regime windows exceed total simulation length. Reduce switches or regime length.")
        gap = available / (self.n_regime_switches + 1)
        cursor = gap
        for _ in range(self.n_regime_switches):
            jitter = 0.0
            if self.jitter_fraction > 0 and gap > 1:
                jitter = rng.uniform(-self.jitter_fraction * gap, self.jitter_fraction * gap)
            length_scale = 1.0
            if self.length_jitter > 0:
                length_scale += rng.uniform(-self.length_jitter, self.length_jitter)
            current_length = int(round(regime_steps * max(length_scale, 0.1)))
            current_length = max(1, min(current_length, total_steps))
            start = int(round(cursor + jitter))
            start = max(0, min(start, total_steps - current_length))
            if intervals and start <= intervals[-1][1]:
                start = intervals[-1][1]
            end = min(start + current_length, total_steps)
            intervals.append((start, end))
            mask[start:end] = 1
            cursor = end + gap
            if end >= total_steps:
                break
        return mask, intervals

    def simulate(
        self,
        total_years: float = 20.0,
        steps_per_year: int = 252 * 24,
        S0: float = 100.0,
        random_state: Optional[int] = None,
    ):
        """Simulate a regime-switching Merton diffusion and return path + metadata."""
        if total_years <= 0:
            raise ValueError("total_years must be positive.")
        if steps_per_year <= 0:
            raise ValueError("steps_per_year must be positive.")
        if S0 <= 0:
            raise ValueError("S0 must be positive.")

        rng = np.random.default_rng(random_state)
        total_steps = int(round(total_years * steps_per_year))
        regime_steps = int(round(self.regime_length_years * steps_per_year))
        regime_steps = max(regime_steps, 1)
        mask, intervals = self._build_mask(total_steps, regime_steps, rng)

        dt = 1.0 / steps_per_year
        times = np.linspace(0.0, total_years, total_steps + 1)
        prices = np.empty(total_steps + 1)
        prices[0] = S0
        regimes = np.zeros(total_steps + 1, dtype=int)
        if mask.size:
            regimes[0] = mask[0]
        log_price = np.log(S0)

        for step in range(total_steps):
            params = self.theta_bear if mask[step] else self.theta_bull
            mu, sigma, lam, gamma, delta = params
            diffusion = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal()
            n_jumps = rng.poisson(lam * dt)
            jump_term = rng.normal(gamma, delta, n_jumps).sum() if n_jumps else 0.0
            log_price += diffusion + jump_term
            prices[step + 1] = np.exp(log_price)
            regimes[step + 1] = mask[step]

        log_returns = np.diff(np.log(prices))
        regime_intervals = [(times[start], times[end]) for start, end in intervals]
        bull_model = MertonJumpDiffusion(self.theta_bull)
        bear_model = MertonJumpDiffusion(self.theta_bear)
        moments = {"bull": bull_model.log_return_moments(dt), "bear": bear_model.log_return_moments(dt)}

        return {
            "times": times,
            "prices": prices,
            "regimes": regimes,
            "log_returns": log_returns,
            "regime_intervals": regime_intervals,
            "dt": dt,
            "moments": moments,
            "theta_bull": self.theta_bull,
            "theta_bear": self.theta_bear,
        }


def merton_log_return_moments(theta: Union[JumpDiffusionParams, Sequence[float]], dt: float) -> Tuple[float, float]:
    """Convenience wrapper to access the analytical log-return moments."""
    return MertonJumpDiffusion(theta).log_return_moments(dt)


def simulate_merton_jump_diffusion(
    T: float = 1.0,
    N: int = 252,
    S0: float = 100.0,
    mu: float = 0.3,
    sigma: float = 0.2,
    lam: float = 5.0,
    gamma: float = -0.05,
    delta: float = 0.1,
    random_state: Optional[int] = None,
):
    """Backwards-compatible function interface for single-regime simulation."""
    model = MertonJumpDiffusion(JumpDiffusionParams(mu, sigma, lam, gamma, delta))
    return model.simulate_path(T=T, N=N, S0=S0, random_state=random_state)


def simulate_merton_jump_regimes(
    total_years: float = 20.0,
    steps_per_year: int = 252 * 24,
    S0: float = 100.0,
    theta_bull: Union[JumpDiffusionParams, Sequence[float]] = JumpDiffusionParams(0.05, 0.2, 5, 0.02, 0.0125),
    theta_bear: Union[JumpDiffusionParams, Sequence[float]] = JumpDiffusionParams(-0.05, 0.4, 10, -0.04, 0.1),
    n_regime_switches: int = 10,
    regime_length_years: float = 0.5,
    length_jitter: float = 0.5,
    jitter_fraction: float = 0.5,
    random_state: Optional[int] = None,
):
    """Backwards-compatible wrapper for the regime-switching simulator."""
    simulator = RegimeSwitchingMerton(
        theta_bull=theta_bull,
        theta_bear=theta_bear,
        n_regime_switches=n_regime_switches,
        regime_length_years=regime_length_years,
        length_jitter=length_jitter,
        jitter_fraction=jitter_fraction,
    )
    return simulator.simulate(total_years=total_years, steps_per_year=steps_per_year, S0=S0, random_state=random_state)


@dataclass
class _BenchmarkStats:
    total: float
    regime_on: float
    regime_off: float
    runtime: float

    def __str__(self) -> str:
        total_pct = f"{self.total * 100:.2f}%"
        on_pct = f"{self.regime_on * 100:.2f}%"
        off_pct = f"{self.regime_off * 100:.2f}%"
        return f"total={total_pct}, on={on_pct}, off={off_pct}, runtime={self.runtime:.3f}s"


class MertonBenchmark:
    """Utility to reproduce the accuracy table on regime-switching Merton paths."""

    def __init__(
        self,
        n_runs: int = 50,
        window: int = 72,
        step: int = 12,
        total_years: float = 20.0,
        steps_per_year: int = 252 * 24,
        theta_bull: Union[JumpDiffusionParams, Sequence[float]] = JumpDiffusionParams(0.05, 0.2, 5, 0.02, 0.0125),
        theta_bear: Union[JumpDiffusionParams, Sequence[float]] = JumpDiffusionParams(-0.05, 0.4, 10, -0.04, 0.1),
        n_regime_switches: int = 10,
        regime_length_years: float = 0.5,
        length_jitter: float = 0.5,
        jitter_fraction: float = 0.5,
        random_state: Optional[int] = None,
        algorithms: Optional[Dict[str, Callable[[Optional[int]], object]]] = None,
        algorithm_windows: Optional[Dict[str, Tuple[int, int]]] = None,
    ):
        self.n_runs = n_runs
        self.window = window
        self.step = step
        self.total_years = total_years
        self.steps_per_year = steps_per_year
        self.theta_bull = theta_bull
        self.theta_bear = theta_bear
        self.n_regime_switches = n_regime_switches
        self.regime_length_years = regime_length_years
        self.length_jitter = length_jitter
        self.jitter_fraction = jitter_fraction
        self.random_state = random_state
        self.algorithms = algorithms
        self.algorithm_windows = algorithm_windows or {}

    @staticmethod
    def _accuracy(pred: np.ndarray, truth: np.ndarray) -> float:
        if truth.size == 0:
            return float("nan")
        return float(np.mean(pred == truth))

    @staticmethod
    def _ci_width(samples: np.ndarray, z: float = 1.96) -> float:
        if len(samples) <= 1:
            return 0.0
        return z * samples.std(ddof=1) / np.sqrt(len(samples))

    @staticmethod
    def _format(mean: float, ci: float, pct: bool = True) -> str:
        if pct:
            return f"{mean * 100:.2f}% ± {ci * 100:.2f}%"
        return f"{mean:.2f}s ± {ci:.2f}s"

    @staticmethod
    def _default_algorithms() -> Dict[str, Callable[[Optional[int]], object]]:
        return {
            "Wasserstein": lambda seed: WassersteinKMeans(n_clusters=2, p_dim=2, max_iter=500, random_state=seed),
            "Moment": lambda seed: MomentKMeans(n_clusters=2, p_dim=2, max_iter=500, random_state=seed),
            "HMM": lambda seed: HMMClustering(n_states=2, random_state=seed, covariance_type="full"),
        }

    @staticmethod
    def _segment(series: pd.Series, window: int, step: int):
        from .utils import segment_time_series

        return segment_time_series(series, window=window, step=step)

    def _algo_window_step(self, name: str) -> Tuple[int, int]:
        if name in self.algorithm_windows:
            win, stp = self.algorithm_windows[name]
            return int(win), int(stp)
        return self.window, self.step

    def run(self, return_details: bool = False):
        rng = np.random.default_rng(self.random_state)
        algorithms = self.algorithms or self._default_algorithms()
        stats: Dict[str, List[_BenchmarkStats]] = {name: [] for name in algorithms}
        details: List[Dict[str, object]] = []

        sim_kwargs = dict(
            total_years=self.total_years,
            steps_per_year=self.steps_per_year,
            theta_bull=self.theta_bull,
            theta_bear=self.theta_bear,
            n_regime_switches=self.n_regime_switches,
            regime_length_years=self.regime_length_years,
            length_jitter=self.length_jitter,
            jitter_fraction=self.jitter_fraction,
        )

        for _ in range(self.n_runs):
            sim_seed = int(rng.integers(0, 2**31 - 1))
            sim = simulate_merton_jump_regimes(random_state=sim_seed, **sim_kwargs)
            prices = pd.Series(sim["prices"], index=sim["times"]).dropna()
            returns = pd.Series(sim["log_returns"], index=sim["times"][1:]).dropna()
            regime_series = pd.Series(sim["regimes"], index=sim["times"])
            segment_cache: Dict[Tuple[int, int], pd.Series] = {}

            def _get_segments(window: int, step: int) -> pd.Series:
                key = (window, step)
                if key not in segment_cache:
                    segment_cache[key] = self._segment(returns, window=window, step=step)
                return segment_cache[key]

            default_segments = _get_segments(self.window, self.step)
            truth_default = regime_series.reindex(default_segments.index).astype(int)
            record = {
                "prices": prices,
                "segments": {"Truth": default_segments},
                "truth": truth_default,
                "predictions": {},
                "regime_intervals": sim["regime_intervals"],
            }

            for algo_name, factory in algorithms.items():
                algo_seed = int(rng.integers(0, 2**31 - 1))
                model = factory(algo_seed)
                algo_window, algo_step = self._algo_window_step(algo_name)
                segments = _get_segments(algo_window, algo_step)
                record["segments"][algo_name] = segments
                using_hmm = isinstance(model, HMMClustering)
                fit_input = returns if using_hmm else segments
                start = perf_counter()
                result = model.fit(fit_input)
                runtime = perf_counter() - start
                labels = getattr(result, "labels", result)
                if using_hmm:
                    if isinstance(labels, pd.Series):
                        pred_series = labels.reindex(segments.index)
                    else:
                        label_index = returns.index[: len(labels)]
                        pred_series = pd.Series(labels, index=label_index)
                        pred_series = pred_series.reindex(segments.index)
                    pred_series = pred_series.ffill()
                    if pred_series.isna().any():
                        raise ValueError(f"HMM predictions cannot be aligned with segment windows for {algo_name}.")
                    pred_series = pred_series.astype(int)
                elif isinstance(labels, pd.Series):
                    pred_series = labels.reindex(segments.index).astype(int)
                else:
                    pred_series = pd.Series(np.asarray(labels, dtype=int), index=segments.index, dtype=int)

                pred_series = pred_series.sort_index()
                truth_series = regime_series.reindex(pred_series.index).astype(float)
                comparison = pd.DataFrame({"pred": pred_series, "truth": truth_series}).dropna()
                if comparison.empty:
                    total_acc = float("nan")
                    on_acc = float("nan")
                    off_acc = float("nan")
                else:
                    preds_arr = comparison["pred"].to_numpy(dtype=int)
                    truth_arr = comparison["truth"].to_numpy(dtype=int)
                    total_acc = self._accuracy(preds_arr, truth_arr)
                    on_mask = truth_arr == 1
                    off_mask = truth_arr == 0
                    on_acc = self._accuracy(preds_arr[on_mask], truth_arr[on_mask])
                    off_acc = self._accuracy(preds_arr[off_mask], truth_arr[off_mask])
                stats[algo_name].append(
                    _BenchmarkStats(
                        total=total_acc,
                        regime_on=on_acc,
                        regime_off=off_acc,
                        runtime=runtime,
                    )
                )
                record["predictions"][algo_name] = pred_series
            details.append(record)
        rows = []
        for name, samples in stats.items():
            totals = np.array([s.total for s in samples])
            ons = np.array([s.regime_on for s in samples])
            offs = np.array([s.regime_off for s in samples])
            times = np.array([s.runtime for s in samples])
            rows.append(
                {
                    "Algorithm": name,
                    "Total": self._format(totals.mean(), self._ci_width(totals)),
                    "Regime-on": self._format(ons.mean(), self._ci_width(ons)),
                    "Regime-off": self._format(offs.mean(), self._ci_width(offs)),
                    "Runtime": self._format(times.mean(), self._ci_width(times), pct=False),
                }
            )

        table = pd.DataFrame(rows).set_index("Algorithm")
        if return_details:
            return table, details, stats
        return table
