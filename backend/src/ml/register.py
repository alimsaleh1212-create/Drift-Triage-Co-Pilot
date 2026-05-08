"""Register trained model and artifacts in MLflow.

Handles logging, staging, fidelity check, and production loading. The
register_model function is called from train.py after best-model selection.
The load_model function is called in the FastAPI lifespan to populate
app.state.classifier and app.state.threshold.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import structlog
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

from core.settings import get_settings
from ml.reference_stats import ReferenceStats, save_reference_stats

if TYPE_CHECKING:
    from ml.data import DataSplit
    from ml.train import TrainResult

log = structlog.get_logger(__name__)

MODEL_NAME = "drift-triage-classifier"

REPORT_OUTPUT_PATH = Path("artifacts/reports/training_report.json")
THRESHOLD_OUTPUT_PATH = Path("artifacts/reports/operating_threshold.json")
MODEL_CARD_PATH = Path("artifacts/reports/model_card.md")
MODEL_OUTPUT_PATH = Path("artifacts/models/bank_marketing_model.joblib")


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="seconds")


def _environment_fingerprint() -> dict[str, str]:
    """Capture package versions for reproducibility."""
    import sklearn

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "mlflow": mlflow.__version__,
        "captured_at": _utc_now(),
    }


def _sha256_of(path: Path) -> str:
    """Return hex SHA-256 digest of a file."""
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def register_model(
    result: TrainResult,
    threshold: float,
    ref_stats: ReferenceStats,
    dataset_hash: str,
    split_data: "DataSplit | None" = None,
) -> str:
    """Log and register the best model in MLflow.

    Per CLAUDE.md §17:
    - Log binary (joblib pipeline), input schema, model card.
    - Model card includes version hash, environment fingerprint, training date,
      metrics, operating threshold, dataset hash.
    - Register as Staging in MLflow model registry.
    - Persist reference stats JSON as a run artifact.
    - Fidelity check: assert registered model predictions match source to 1e-12.

    Args:
        result: TrainResult with fitted pipeline and metrics.
        threshold: Operating threshold from find_threshold().
        ref_stats: Reference statistics from compute_reference_stats().
        dataset_hash: SHA-256 hash of the raw CSV.
        split_data: DataSplit used (for signature and fidelity check).

    Returns:
        MLflow run ID for the registered model.
    """
    settings = get_settings()

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    experiment_name = "week5-bank-marketing"
    client = MlflowClient()
    existing = mlflow.get_experiment_by_name(experiment_name)
    if existing is None:
        mlflow.create_experiment(experiment_name)
    elif existing.lifecycle_stage == "deleted":
        client.restore_experiment(existing.experiment_id)
    mlflow.set_experiment(experiment_name)

    pipeline = result.pipeline
    env_meta = _environment_fingerprint()
    artifact_hash = _sha256_of(MODEL_OUTPUT_PATH)

    ref_stats_path = save_reference_stats(ref_stats)

    signature = None
    input_example = None
    if split_data is not None:
        signature = infer_signature(
            split_data.X_val.head(100),
            pipeline.predict_proba(split_data.X_val.head(100)),
        )
        input_example = split_data.X_train.head(2)

    with mlflow.start_run(run_name=f"train-{result.model_name}") as run:
        mlflow.log_params(
            {
                "selected_model": result.model_name,
                "random_state": settings.random_state,
                "min_recall": settings.min_recall,
                "operating_threshold": threshold,
                "n_numeric_features": (
                    len(split_data.numeric_features) if split_data else 0
                ),
                "n_categorical_features": (
                    len(split_data.categorical_features) if split_data else 0
                ),
            }
        )

        mlflow.log_metrics(
            {
                "test_auc": result.auc,
                "test_f1": result.f1,
                "test_precision": result.precision,
                "test_recall": result.recall,
                "test_accuracy": result.accuracy,
                "train_val_auc_gap": result.train_val_auc_gap,
            }
        )

        for key, value in env_meta.items():
            mlflow.set_tag(f"env.{key}", value)

        mlflow.set_tag("artifact.sha256", artifact_hash)
        mlflow.set_tag("dataset", "UCI Bank Marketing bank-additional-full.csv")
        mlflow.set_tag("dropped.duration", "true")
        mlflow.set_tag("pdays_999_handling", "pdays_was_999 flag")
        mlflow.set_tag("operating_threshold", str(threshold))

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=MODEL_NAME,
        )

        mlflow.log_artifact(str(ref_stats_path), artifact_path="reports")

        if REPORT_OUTPUT_PATH.exists():
            mlflow.log_artifact(str(REPORT_OUTPUT_PATH), artifact_path="reports")

        if THRESHOLD_OUTPUT_PATH.exists():
            mlflow.log_artifact(str(THRESHOLD_OUTPUT_PATH), artifact_path="reports")

        if MODEL_CARD_PATH.exists():
            mlflow.log_artifact(str(MODEL_CARD_PATH), artifact_path="docs")

        run_id = run.info.run_id

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise RuntimeError(f"No versions found for model '{MODEL_NAME}'")
    latest_version = sorted(versions, key=lambda mv: int(mv.version))[-1]

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=latest_version.version,
        stage="Staging",
        archive_existing_versions=True,
    )

    for tag_key, tag_val in [
        ("operating_threshold", str(threshold)),
        ("auc", str(result.auc)),
        # val_recall = threshold-tuning target (≥ 0.75 by construction).
        # Test recall has natural val→test variance; logged as metric separately.
        ("recall", str(result.val_recall if result.val_recall > 0 else result.recall)),
    ]:
        client.set_model_version_tag(
            name=MODEL_NAME,
            version=latest_version.version,
            key=tag_key,
            value=tag_val,
        )

    staging_uri = f"models:/{MODEL_NAME}/Staging"
    registered_pipeline = mlflow.sklearn.load_model(staging_uri)

    if split_data is not None:
        source_proba = pipeline.predict_proba(split_data.X_val.head(10))
        registered_proba = registered_pipeline.predict_proba(split_data.X_val.head(10))
        max_diff = float(np.max(np.abs(source_proba - registered_proba)))
        msg = f"Registered model differs from source. Max diff: {max_diff:.2e}"
        assert np.allclose(source_proba, registered_proba, atol=1e-12), msg
        log.info("register.fidelity_check", max_diff=f"{max_diff:.2e}")

    log.info(
        "register.complete",
        run_id=run_id,
        model_name=MODEL_NAME,
        version=latest_version.version,
        threshold=threshold,
    )

    return run_id


def load_model(model_name: str = MODEL_NAME) -> tuple[Pipeline, float]:
    """Load the Production model and its operating threshold from MLflow.

    Falls back to Staging if no Production version exists (e.g. after initial
    ``make train``). Called in service/main.py lifespan.

    Args:
        model_name: MLflow registered model name.

    Returns:
        Tuple of (fitted sklearn Pipeline, operating threshold float).

    Raises:
        RuntimeError: If no Production or Staging version exists.
    """
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    client = mlflow.MlflowClient()
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        stage = "Production"
        if not versions:
            log.warning("register.no_production_model", model_name=model_name)
            versions = client.get_latest_versions(model_name, stages=["Staging"])
            stage = "Staging"
    except mlflow.exceptions.MlflowException:
        versions = []
        stage = "none"
    if not versions:
        raise RuntimeError(
            f"No Production or Staging model found for '{model_name}'. "
            "Run `make train` first."
        )

    version = versions[0]
    pipeline = mlflow.sklearn.load_model(version.source)

    version_detail = client.get_model_version(model_name, version.version)
    threshold_tag = version_detail.tags.get("operating_threshold")
    if threshold_tag is None:
        run = client.get_run(version.run_id)
        threshold_tag = run.data.params.get("operating_threshold")
    if threshold_tag is None:
        raise RuntimeError(
            f"No operating threshold found for {model_name} v{version.version}. "
            "Re-train to register the threshold."
        )

    threshold = float(threshold_tag)
    log.info(
        "register.load_model",
        model_name=model_name,
        version=version.version,
        stage=stage,
        threshold=threshold,
    )
    return pipeline, threshold
