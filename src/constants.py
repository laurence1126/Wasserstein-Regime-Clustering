"""Shared constants for visualizations and cluster styling."""

from typing import Dict

# Central colour palette for cluster/regime plots; extend as needed.
CLUSTER_PALETTE: Dict[int, str] = {
    0: "#4477AA",
    1: "#228833",
    2: "#EE6677",
    3: "#CCBB44",
    4: "#66CCEE",
    5: "#AA3377",
    6: "#BBBBBB",
}

__all__ = ["CLUSTER_PALETTE"]
