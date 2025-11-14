# `src/clusterings.py`

Utilities for clustering univariate return windows.

## `WassersteinKMeans`
- Clusters empirical distributions using Wasserstein distance (`p=1` or `p=2`).
- Key parameters: `n_clusters`, `p`, `max_iter`, `tol`, `random_state`.
- `fit(segments, initial_centroids=None)`: accepts either a list of numpy arrays or a pandas Series of segments (values are arrays, index is timestamp). Supports warm starts via `initial_centroids`. Returns `WKMeansResult`.
- `predict(segments)`: accepts the same inputs (list or Series). If a Series is provided the function returns a Series of labels aligned to the input index; otherwise it returns a numpy array.
- `plot_centroids_cdf(title)`: plots centroid CDFs for visualization.

## `MomentKMeans`
- Traditional K-means applied to raw moments of segments.
- Parameters: `n_clusters`, `p_moments`, `standardize`, `max_iter`, `tol`, `init` (`kmeans++` or `random`).
- Methods mirror scikit-learn: `fit`, `predict`, `transform` (returns moment features), `plot_moment_space`.

These classes are used by the regime strategy to partition return windows and derive regime labels.
