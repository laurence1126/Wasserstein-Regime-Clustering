# `src/clustering_eval.py`

Implements Maximum Mean Discrepancy (MMD^2) utilities and classic clustering metrics for comparing return-window distributions.

## `MMDCalculator`
- Parameters: `sigma` (bandwidth), `unbiased` (choose U- vs V-statistic), `random_state` for reproducibility.
- `_median_sigma`: median heuristic when `sigma` is None.
- `mmd2(X, Y, sigma=None)`: computes Gaussian RBF MMD^2 between two sets of equal-length segments.
- `bootstrap_between(X, Y, B, m_per_group, replace)`: bootstraps MMD^2 between two clusters by sampling `m` segments from each cluster `B` times.
- `bootstrap_within(X, B, m_per_half, replace)`: bootstraps within-cluster MMD^2 by randomly splitting observations into two halves.
- `_largest_two_clusters(segments, labels)`: helper to extract the two most populated clusters.
- `compare_two_clusterings_hist(...)`: histogram comparison of between-cluster MMD^2 distributions for two labeling schemes (e.g., Wasserstein vs. Moment).
- `plot_within_two_methods(...)`: side-by-side histograms of within-cluster MMD^2 for the two largest clusters under two labelings.

## `ClusteringMetrics`
- `davies_bouldin_index`: Davies-Bouldin score per Definition 3.3.
- `dunn_index`: Dunn index using intra/inter cluster distances from Definition 3.4.
- `silhouette_scores`: point-wise silhouette coefficients from Definition 3.5.
- `alpha_silhouette`: α-average silhouette described in Remark 3.6 (α ∈ (0, 1]).
- `evaluate_all`: convenience helper returning a dictionary with every metric.

Use these tools to quantify how separable regimes are (between-cluster MMD) or how stable each regime's distribution is (within-cluster MMD). See `jupyter/clustering_eval_examples.ipynb` for a complete walkthrough that reproduces the evaluation figures/tables independently of the clustering tutorial. EOF
