"""Threshold tuning: find highest threshold where recall >= min_recall.

The operating threshold is stored with the model in MLflow and used at
serving time to convert probabilities into binary predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline

import structlog

from core.settings import get_settings

log = structlog.get_logger(__name__)


def find_threshold(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    min_recall: float | None = None,
) -> float:
    """Find highest decision threshold where recall >= min_recall on X_val.

    Per CLAUDE.md §17: find the highest threshold such that recall >= 0.75
    on the validation set. If no threshold achieves the minimum recall,
    fall back to the F1-optimal threshold.

    Args:
        pipeline: Fitted sklearn pipeline with predict_proba support.
        X_val: Validation features.
        y_val: Validation labels (0/1).
        min_recall: Minimum acceptable recall. Defaults to Settings.min_recall.

    Returns:
        Float threshold in (0, 1). Applied to predict_proba[:, 1] at serve time.

    Raises:
        ValueError: If y_val is empty.
    """
    if min_recall is None:
        min_recall = get_settings().min_recall

    proba = pipeline.predict_proba(X_val)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_val, proba)

    precisions_for_thresholds = precisions[:-1]
    recalls_for_thresholds = recalls[:-1]

    f1_scores = (
        2 * precisions_for_thresholds * recalls_for_thresholds
    ) / np.where(
        (precisions_for_thresholds + recalls_for_thresholds) > 0,
        precisions_for_thresholds + recalls_for_thresholds,
        1.0,
    )

    best_f1_idx = int(np.argmax(f1_scores))
    f1_optimal_threshold = float(thresholds[best_f1_idx])

    recall_mask = recalls_for_thresholds >= min_recall

    if recall_mask.any():
        operating_threshold = float(thresholds[recall_mask].max())
    else:
        operating_threshold = f1_optimal_threshold
        log.warning(
            "threshold.fallback",
            min_recall=min_recall,
            best_recall=float(recalls_for_thresholds.max()),
            fallback_threshold=operating_threshold,
        )

    log.info(
        "threshold.found",
        operating_threshold=operating_threshold,
        min_recall=min_recall,
    )

    return operating_threshold