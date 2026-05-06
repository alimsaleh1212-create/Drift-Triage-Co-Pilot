"""Compute and persist reference statistics for drift detection.

At training time, compute per-feature distribution statistics and predicted
class proportions on the test set. Persist to JSON for loading at serve time.

Per CLAUDE.md §17:
- Numeric: mean, std, and decile quantiles.
- Categorical: relative frequency of each category.
- Output: predicted class proportions (class 0 / class 1).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

import structlog

from ml.data import DataSplit

log = structlog.get_logger(__name__)

REFERENCE_STATS_PATH = Path("artifacts/reference_stats.json")


@dataclass
class ReferenceStats:
    """Distribution statistics computed on the test split.

    Used at serve time to compute PSI (numeric) and chi-squared (categorical).
    """

    numeric: dict[str, dict[str, Any]]
    categorical: dict[str, dict[str, float]]
    output_proportions: dict[str, float]
    dataset_hash: str


def compute_reference_stats(
    pipeline: Pipeline,
    split: DataSplit,
) -> ReferenceStats:
    """Compute per-feature distribution stats and output proportions.

    Args:
        pipeline: Fitted sklearn pipeline (used to predict on test set).
        split: DataSplit with train/val/test splits and feature lists.

    Returns:
        ReferenceStats ready for persistence via save_reference_stats().
    """
    X_test = split.X_test
    y_test = split.y_test

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    numeric_stats: dict[str, dict[str, Any]] = {}
    for feature in split.numeric_features:
        if feature not in X_test.columns:
            continue
        series = X_test[feature].astype(float)
        numeric_stats[feature] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "quantiles": [float(q) for q in np.quantile(series, np.arange(0.1, 1.0, 0.1))],
        }

    categorical_stats: dict[str, dict[str, float]] = {}
    for feature in split.categorical_features:
        if feature not in X_test.columns:
            continue
        counts = X_test[feature].value_counts(normalize=True).to_dict()
        categorical_stats[feature] = {str(k): float(v) for k, v in counts.items()}

    unique, counts = np.unique(y_pred, return_counts=True)
    proportions = dict(zip([str(u) for u in unique], [float(c) / len(y_pred) for c in counts]))
    output_proportions: dict[str, float] = {"0": proportions.get("0", 0.0), "1": proportions.get("1", 0.0)}

    log.info(
        "reference_stats.computed",
        numeric_features=len(numeric_stats),
        categorical_features=len(categorical_stats),
    )

    return ReferenceStats(
        numeric=numeric_stats,
        categorical=categorical_stats,
        output_proportions=output_proportions,
        dataset_hash=split.dataset_hash,
    )


def save_reference_stats(
    ref_stats: ReferenceStats,
    path: Path = REFERENCE_STATS_PATH,
) -> Path:
    """Persist reference statistics to JSON.

    Args:
        ref_stats: Statistics computed by compute_reference_stats().
        path: Output path. Defaults to artifacts/reference_stats.json.

    Returns:
        Path to the written file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "numeric": ref_stats.numeric,
        "categorical": ref_stats.categorical,
        "output_proportions": ref_stats.output_proportions,
        "dataset_hash": ref_stats.dataset_hash,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)

    log.info("reference_stats.saved", path=str(path))
    return path


def load_reference_stats(path: Path = REFERENCE_STATS_PATH) -> ReferenceStats:
    """Load persisted reference stats from JSON artifact.

    Called in service/main.py lifespan to populate app.state.ref_stats.

    Args:
        path: Path to the JSON artifact written by save_reference_stats().

    Returns:
        ReferenceStats ready for use by drift detection.

    Raises:
        FileNotFoundError: If artifact missing — run ``make train`` first.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Reference stats not found at {path}. Run `make train` first."
        )
    with path.open() as f:
        data = json.load(f)
    return ReferenceStats(**data)


def _json_default(obj: Any) -> Any:
    """Convert numpy types for JSON serialisation."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")