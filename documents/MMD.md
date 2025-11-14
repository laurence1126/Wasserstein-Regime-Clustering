# `src/MMD.py`

Implements Maximum Mean Discrepancy (MMD^2) utilities for comparing return-window distributions.

## `MMDCalculator`
- Parameters: `sigma` (bandwidth), `unbiased` (choose U- vs V-statistic), `random_state` for reproducibility.
- `_median_sigma`: median heuristic when `sigma` is None.
- `mmd2(X, Y, sigma=None)`: computes Gaussian RBF MMD^2 between two sets of equal-length segments.
- `bootstrap_between(X, Y, B, m_per_group, replace)`: bootstraps MMD^2 between two clusters by sampling `m` segments from each cluster `B` times.
- `bootstrap_within(X, B, m_per_half, replace)`: bootstraps within-cluster MMD^2 by randomly splitting observations into two halves.
- `_largest_two_clusters(segments, labels)`: helper to extract the two most populated clusters.
- `compare_two_clusterings_hist(...)`: histogram comparison of between-cluster MMD^2 distributions for two labeling schemes (e.g., Wasserstein vs. Moment).
- `plot_within_two_methods(...)`: side-by-side histograms of within-cluster MMD^2 for the two largest clusters under two labelings.

Use these tools to quantify how separable regimes are (between-cluster MMD) or how stable each regime's distribution is (within-cluster MMD). EOF
