"""Train candidate classifiers and select the best by AUC.

STUB — partner implements the body. Signatures and return types are final.

Usage (after partner fills in):
    uv run python -m ml.train
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]

from ml.data import DataSplit


@dataclass
class TrainResult:
    """Best model and its evaluation metrics."""

    pipeline: Pipeline
    model_name: str
    auc: float
    recall: float
    precision: float
    f1: float
    cv_auc_mean: float
    cv_auc_std: float


def train(split: DataSplit) -> TrainResult:
    """Train LogReg, RandomForest, GradientBoosting + DummyClassifier baseline.

    Partner implements this function. Requirements per CLAUDE.md §17:
    - StratifiedKFold(k=5) cross-validation for model selection.
    - Pick best by AUC on the validation set.
    - All stochastic calls use random_state from Settings.

    Args:
        split: Stratified data split from data.load_data().

    Returns:
        TrainResult with the best fitted pipeline and its metrics.
    """
    raise NotImplementedError("ML stub — partner implements")


if __name__ == "__main__":
    from ml.data import load_data
    from ml.reference_stats import compute_reference_stats
    from ml.register import register_model
    from ml.threshold import find_threshold

    split = load_data()
    result = train(split)
    threshold = find_threshold(result.pipeline, split.X_val, split.y_val)
    ref_stats = compute_reference_stats(result.pipeline, split)
    register_model(result, threshold, ref_stats, split.dataset_hash)
