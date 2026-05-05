"""Threshold tuning: find highest threshold meeting recall >= min_recall.

STUB — partner implements the body. Signatures and return types are final.
"""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]

from core.settings import get_settings


def find_threshold(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    min_recall: float | None = None,
) -> float:
    """Find highest decision threshold where recall >= min_recall on X_val.

    Partner implements this function. Per CLAUDE.md §17: find the highest
    threshold such that recall >= 0.75 on the validation set. This threshold
    is stored in MLflow alongside the model and used at serving time.

    Args:
        pipeline: Fitted sklearn pipeline with predict_proba support.
        X_val: Validation features.
        y_val: Validation labels.
        min_recall: Minimum acceptable recall. Defaults to Settings.min_recall.

    Returns:
        Float threshold in (0, 1). Applied to predict_proba[:, 1] at serve time.

    Raises:
        ValueError: If no threshold achieves min_recall (model is too weak).
    """
    raise NotImplementedError("ML stub — partner implements")
