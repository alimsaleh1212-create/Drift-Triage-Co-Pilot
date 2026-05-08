"""Prediction helpers: batch inference and feature engineering.

Per CLAUDE.md §6, the pipeline and threshold are injected via FastAPI
Depends() from app.state — no globals, no module-level joblib.load().
This module only provides pure functions that take the pipeline and threshold
as arguments.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import structlog
from sklearn.pipeline import Pipeline

log = structlog.get_logger(__name__)


def prepare_prediction_features(feature_dict: dict[str, Any]) -> dict[str, Any]:
    """Apply feature engineering rules matching the training pipeline.

    Adds the ``pdays_was_999`` binary flag derived from the ``pdays`` column.
    Callers should pass ``payload.model_dump(by_alias=True)`` as input.

    Args:
        feature_dict: Serialised feature dictionary (aliases expanded).

    Returns:
        The same dictionary with ``pdays_was_999`` added.
    """
    feature_dict["pdays_was_999"] = int(feature_dict["pdays"] == 999)
    return feature_dict


def predict_batch(
    X: pd.DataFrame,
    pipeline: Pipeline,
) -> np.ndarray:
    """Predict positive-class probabilities for a batch of rows.

    Used by drift detection (output drift), reference stats, and the
    replay-test worker.  The function is synchronous (sklearn is CPU-bound);
    callers wrap it in ``asyncio.to_thread()`` at the router level.

    Args:
        X: DataFrame with columns matching training features.
        pipeline: Fitted sklearn pipeline (injected via Depends or loaded).

    Returns:
        1-D numpy array of probabilities for the positive class.
    """
    return pipeline.predict_proba(X)[:, 1]
