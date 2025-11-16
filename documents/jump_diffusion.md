# `src/jump_diffusion.py`

Utilities focused on Merton jump-diffusion models used for synthetic experiments.

- `JumpDiffusionParams`: typed tuple `(mu, sigma, lam, gamma, delta)` describing drift, diffusion, jump intensity, and jump-size distribution.
- `MertonJumpDiffusion`: class encapsulating a single regime. Methods:
  - `simulate_path(T, N, S0, random_state)`: Euler-style path generator returning `(times, prices, jumps)`.
  - `log_return_moments(dt)`: closed-form mean/variance of log-returns over mesh `dt` (equations (39)–(40)).
- `RegimeSwitchingMerton`: wraps the bull/bear experiment from Section 3.3.2. Configure with `(θ_bull, θ_bear, n_regime_switches, regime_length_years, length_jitter, jitter_fraction)` and call `simulate(total_years, steps_per_year, S0, random_state)` to obtain the price path, regime mask, log-returns, highlighted intervals, and theoretical moments. `length_jitter` perturbs each bear-window duration to avoid overly periodic regimes.
- `MertonBenchmark`: repeats the regime-switching simulation, segments returns, runs the configured clustering algorithms (default: Wasserstein/Moment K-means), and returns a DataFrame with mean ± 95% CI for total/regime-on/regime-off accuracy and runtime. `run(return_details=True)` also yields the simulated price Series, segment Series, true regime Series, and per-algorithm label Series so other plotting utilities (e.g., `plot_regimes_over_price`, `scatter_mean_variance`) can be reused. The helper `run_merton_benchmark(...)` instantiates the class with keyword arguments.
- `simulate_merton_jump_diffusion(...)` and `simulate_merton_jump_regimes(...)`: backwards-compatible function wrappers that delegate to the corresponding classes—handy for existing notebooks that imported them from `utils`. See `jupyter/jump_diffusion_compare.ipynb` for an end-to-end example that calls `MertonBenchmark.run(return_details=True)` and then feeds the returned Series into `plot_regimes_over_price` / `scatter_mean_variance`.

Import these helpers to keep jump-diffusion logic (including the benchmark) separate from the generic utilities module.
