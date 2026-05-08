"""Train candidate classifiers and select the best by F1 at operating threshold.

Writes artifacts to ``artifacts/``:
- ``models/bank_marketing_model.joblib`` — fitted pipeline + metadata
- ``reports/training_report.json`` — full model comparison and metrics
- ``reports/operating_threshold.json`` — threshold config
- ``reports/model_card.md`` — human-readable model card

Usage::

    cd backend
    uv run python -m ml.train
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import structlog
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from core.settings import Settings, get_settings
from ml.data import DataSplit, load_data
from ml.pipeline import build_pipeline
from ml.reference_stats import compute_reference_stats
from ml.register import MODEL_NAME, register_model
from ml.threshold import find_threshold

log = structlog.get_logger(__name__)

CANDIDATE_MODELS = ["dummy", "logistic_regression", "random_forest", "extra_trees"]

MODEL_OUTPUT_PATH = Path("artifacts/models/bank_marketing_model.joblib")
REPORT_OUTPUT_PATH = Path("artifacts/reports/training_report.json")
THRESHOLD_OUTPUT_PATH = Path("artifacts/reports/operating_threshold.json")
MODEL_CARD_PATH = Path("artifacts/reports/model_card.md")


def _to_jsonable(obj: Any) -> Any:
    """Convert numpy/pandas objects to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _sha256_of(path: Path) -> str:
    """Return hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class TrainResult:
    """Best model, its evaluation metrics, and full comparison data."""

    pipeline: Pipeline
    model_name: str
    auc: float
    recall: float
    precision: float
    f1: float
    accuracy: float
    threshold: float
    val_recall: float = 0.0  # recall on val set at operating threshold (what was tuned)
    all_results: dict = field(default_factory=dict)
    split_info: dict = field(default_factory=dict)
    dataset_hash: str = ""
    train_val_auc_gap: float = 0.0


def evaluate_with_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Evaluate model probabilities using a chosen decision threshold.

    Args:
        y_true: Ground truth labels (0/1).
        y_proba: Predicted probabilities for the positive class.
        threshold: Decision threshold.

    Returns:
        Dict with accuracy, precision, recall, f1, roc_auc, threshold.
    """
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def train(split: DataSplit) -> TrainResult:
    """Train candidate models, select best by F1 at operating threshold.

    Per CLAUDE.md §17: evaluates dummy baseline + 3 classifiers, picks the
    best by validation F1 at the operating threshold chosen by
    ``find_threshold()`` (recall >= min_recall).

    Args:
        split: Stratified data split from ``load_data()``.

    Returns:
        TrainResult with best fitted pipeline, all candidate results,
        and split metadata.
    """
    settings = get_settings()
    results: dict[str, dict] = {}

    for model_name in CANDIDATE_MODELS:
        log.info("train.start", model=model_name)

        pipeline = build_pipeline(
            model_name=model_name,
            numeric_features=split.numeric_features,
            categorical_features=split.categorical_features,
            random_state=settings.random_state,
        )
        pipeline.fit(split.X_train, split.y_train)

        threshold = find_threshold(pipeline, split.X_val, split.y_val)

        val_proba = pipeline.predict_proba(split.X_val)[:, 1]
        val_metrics = evaluate_with_threshold(split.y_val, val_proba, threshold)

        train_proba = pipeline.predict_proba(split.X_train)[:, 1]
        train_metrics = evaluate_with_threshold(split.y_train, train_proba, threshold)

        train_val_auc_gap = train_metrics["roc_auc"] - val_metrics["roc_auc"]

        results[model_name] = {
            "pipeline": pipeline,
            "threshold": threshold,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "train_val_auc_gap": train_val_auc_gap,
        }

        log.info(
            "train.candidate",
            model=model_name,
            threshold=f"{threshold:.6f}",
            val_auc=f"{val_metrics['roc_auc']:.4f}",
            val_f1=f"{val_metrics['f1']:.4f}",
            val_recall=f"{val_metrics['recall']:.4f}",
        )

    best_name = max(results, key=lambda n: results[n]["val_metrics"]["f1"])
    best = results[best_name]

    test_proba = best["pipeline"].predict_proba(split.X_test)[:, 1]
    test_metrics = evaluate_with_threshold(split.y_test, test_proba, best["threshold"])

    log.info(
        "train.best",
        model=best_name,
        test_auc=f"{test_metrics['roc_auc']:.4f}",
        test_f1=f"{test_metrics['f1']:.4f}",
        test_recall=f"{test_metrics['recall']:.4f}",
    )

    split_info = {
        "train_rows": len(split.X_train),
        "validation_rows": len(split.X_val),
        "test_rows": len(split.X_test),
    }

    return TrainResult(
        pipeline=best["pipeline"],
        model_name=best_name,
        auc=test_metrics["roc_auc"],
        recall=test_metrics["recall"],
        precision=test_metrics["precision"],
        f1=test_metrics["f1"],
        accuracy=test_metrics["accuracy"],
        threshold=best["threshold"],
        val_recall=best["val_metrics"]["recall"],
        all_results=results,
        split_info=split_info,
        dataset_hash=split.dataset_hash,
        train_val_auc_gap=best["train_val_auc_gap"],
    )


def _write_training_report(result: TrainResult, settings: "Settings") -> Path:
    """Write the full training report JSON with all model comparisons."""
    report = {
        "created_at": _utc_now(),
        "random_state": settings.random_state,
        "min_recall": settings.min_recall,
        "split": result.split_info,
        "selected_model": result.model_name,
        "selected_threshold": result.threshold,
        "threshold_rule": (
            f"highest threshold where validation recall >= {settings.min_recall}"
        ),
        "dataset_hash": result.dataset_hash,
        "model_artifact_sha256": _sha256_of(MODEL_OUTPUT_PATH),
        "validation_results": {
            name: {
                "threshold": data["threshold"],
                "train_metrics": data["train_metrics"],
                "val_metrics": data["val_metrics"],
                "train_val_auc_gap": data["train_val_auc_gap"],
            }
            for name, data in result.all_results.items()
        },
        "test_metrics": {
            "auc": result.auc,
            "f1": result.f1,
            "precision": result.precision,
            "recall": result.recall,
            "accuracy": result.accuracy,
        },
        "train_val_auc_gap": result.train_val_auc_gap,
        "dropped_columns": ["duration"],
        "special_handling": {
            "duration": "Dropped because it leaks the target.",
            "pdays": "Added pdays_was_999 flag because pdays=999 is a sentinel value.",
            "unknown": "Kept unknown as a real category.",
        },
    }

    REPORT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(report), f, indent=2)

    log.info("train.report_written", path=str(REPORT_OUTPUT_PATH))
    return REPORT_OUTPUT_PATH


def _write_threshold_config(result: TrainResult, settings: "Settings") -> Path:
    """Write the operating threshold configuration JSON."""
    threshold_artifact = {
        "registered_model_name": MODEL_NAME,
        "selected_model": result.model_name,
        "operating_threshold": result.threshold,
        "rule": f"highest threshold where validation recall >= {settings.min_recall}",
        "created_at": _utc_now(),
    }

    THRESHOLD_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with THRESHOLD_OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(threshold_artifact), f, indent=2)

    log.info("train.threshold_written", path=str(THRESHOLD_OUTPUT_PATH))
    return THRESHOLD_OUTPUT_PATH


def _write_model_card(result: TrainResult, settings: "Settings") -> Path:
    """Write a markdown model card per CLAUDE.md §17 requirements."""
    from ml.register import _environment_fingerprint

    env_meta = _environment_fingerprint()
    env_json = json.dumps(env_meta, indent=2)
    model_name = MODEL_NAME

    card = f"""\
# Model Card — {model_name}

**Status:** Staging candidate
**Created at:** {_utc_now()}
**Selected model:** `{result.model_name}`
**Artifact hash:** `{_sha256_of(MODEL_OUTPUT_PATH)}`

## Intended use

Predict whether a client will subscribe to a term deposit using
the UCI Bank Marketing dataset.

The model outputs a probability for the positive class:

- `yes` = client subscribes
- `no` = client does not subscribe

## Training data

Dataset: UCI Bank Marketing, `bank-additional-full.csv`

Split:

| Split | Rows |
|---|---:|
| Train | {result.split_info.get('train_rows', 'N/A')} |
| Validation | {result.split_info.get('validation_rows', 'N/A')} |
| Test | {result.split_info.get('test_rows', 'N/A')} |

The split is stratified 60/20/20 with `random_state={settings.random_state}`.

## Important data rules

- `duration` is dropped because it leaks the target.
- `pdays == 999` is treated as a sentinel, with an extra `pdays_was_999` flag.
- `"unknown"` values are kept as real categories, not deleted.
- Categorical features are encoded using `OneHotEncoder(handle_unknown="ignore")`.

## Architecture

sklearn `Pipeline`:

1. `ColumnTransformer`
   - Numeric: median imputation + scaling
   - Categorical: most-frequent imputation + one-hot encoding
2. Classifier: `{result.model_name}`

## Operating threshold

Chosen threshold: `{result.threshold}`

Rule: highest threshold where validation recall >= {settings.min_recall}

## Final test metrics

| Metric | Value |
|---|---:|
| AUC | {result.auc:.4f} |
| F1 | {result.f1:.4f} |
| Precision | {result.precision:.4f} |
| Recall | {result.recall:.4f} |
| Accuracy | {result.accuracy:.4f} |

## Generalization check

Train-Val AUC gap for selected model: `{result.train_val_auc_gap:.4f}`

A large gap would suggest overfitting.

## Known limitations

- The positive class is rare, so precision is low when recall is
  forced near {settings.min_recall}.
- The model is trained on historical campaign data and may drift if
  economic indicators or customer behavior shift.
- The threshold is a deployment policy and should be logged with every prediction.

## Environment fingerprint

{env_json}
"""

    MODEL_CARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_CARD_PATH.write_text(card, encoding="utf-8")

    log.info("train.model_card_written", path=str(MODEL_CARD_PATH))
    return MODEL_CARD_PATH


def main() -> None:
    """Entry point: train, evaluate, write artifacts, register model."""
    settings = get_settings()
    np.random.seed(settings.random_state)

    log.info("train.data_loading")
    split = load_data()

    log.info("train.training")
    result = train(split)

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "registered_model_name": MODEL_NAME,
        "selected_model": result.model_name,
        "pipeline": result.pipeline,
        "threshold": result.threshold,
        "numeric_features": split.numeric_features,
        "categorical_features": split.categorical_features,
        "target_mapping": {"no": 0, "yes": 1},
        "created_at": _utc_now(),
    }
    joblib.dump(artifact, MODEL_OUTPUT_PATH)
    log.info("train.artifact_saved", path=str(MODEL_OUTPUT_PATH))

    _write_training_report(result, settings)
    _write_threshold_config(result, settings)
    _write_model_card(result, settings)

    ref_stats = compute_reference_stats(result.pipeline, split)

    log.info("train.registering")
    register_model(
        result=result,
        threshold=result.threshold,
        ref_stats=ref_stats,
        dataset_hash=split.dataset_hash,
        split_data=split,
    )

    log.info(
        "train.complete",
        best_model=result.model_name,
        threshold=f"{result.threshold:.6f}",
        test_auc=f"{result.auc:.4f}",
        test_f1=f"{result.f1:.4f}",
        test_recall=f"{result.recall:.4f}",
    )


if __name__ == "__main__":
    main()
