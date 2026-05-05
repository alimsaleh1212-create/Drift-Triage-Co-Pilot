"""Compute and persist reference statistics for drift detection.

STUB — partner implements the body. Signatures and return types are final.
These stats are loaded at serving time via service/main.py lifespan.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.data import DataSplit

REFERENCE_STATS_PATH = Path("artifacts/reference_stats.json")


@dataclass
class ReferenceStats:
    """Distribution statistics computed on the training test split.

    Used at serving time to compute PSI (numeric) and chi² (categorical).
    """

    # {feature_name: {"mean": float, "std": float, "quantiles": list[float]}}
    numeric: dict[str, dict[str, Any]]
    # {feature_name: {category: float}}  — relative frequencies summing to 1
    categorical: dict[str, dict[str, float]]
    # Predicted class proportions on test set: {0: float, 1: float}
    output_proportions: dict[str, float]
    dataset_hash: str


def compute_reference_stats(
    pipeline: object,
    split: DataSplit,
) -> ReferenceStats:
    """Compute per-feature distribution stats and output proportions.

    Partner implements this function. Per CLAUDE.md §17:
    - Numeric: mean, std, and decile quantiles on the test set.
    - Categorical: relative frequency of each category on the test set.
    - Output: predicted class proportions (class 0 / class 1) on the test set.

    Args:
        pipeline: Fitted sklearn pipeline (used to predict on test set).
        split: DataSplit; stats are computed on the test split.

    Returns:
        ReferenceStats persisted to REFERENCE_STATS_PATH as JSON artifact.
    """
    raise NotImplementedError("ML stub — partner implements")


def load_reference_stats(path: Path = REFERENCE_STATS_PATH) -> ReferenceStats:
    """Load persisted reference stats from JSON artifact.

    Called in service/main.py lifespan to populate app.state.ref_stats.

    Args:
        path: Path to the JSON artifact written by compute_reference_stats.

    Returns:
        ReferenceStats ready for use by drift detection.

    Raises:
        FileNotFoundError: If artifact missing — run `make train` first.
    """
    import json

    if not path.exists():
        raise FileNotFoundError(
            f"Reference stats not found at {path}. Run `make train` first."
        )
    with path.open() as f:
        data = json.load(f)
    return ReferenceStats(**data)
