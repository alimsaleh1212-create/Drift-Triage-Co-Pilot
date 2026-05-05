"""Register trained model and artifacts in MLflow.

STUB — partner implements the body. Signatures and return types are final.
"""

from __future__ import annotations

from ml.reference_stats import ReferenceStats
from ml.train import TrainResult


def register_model(
    result: TrainResult,
    threshold: float,
    ref_stats: ReferenceStats,
    dataset_hash: str,
) -> str:
    """Log and register the best model in MLflow.

    Partner implements this function. Per CLAUDE.md §17:
    - Log binary (joblib pipeline), input schema, model card.
    - Model card includes: version hash, env fingerprint, training date,
      metrics, operating threshold, dataset hash.
    - Register as 'Staging' in MLflow model registry.
    - Persist reference stats JSON as a run artifact.

    Args:
        result: TrainResult with fitted pipeline and metrics.
        threshold: Operating threshold from find_threshold().
        ref_stats: Reference statistics from compute_reference_stats().
        dataset_hash: SHA-256 hash of the raw CSV.

    Returns:
        MLflow run ID for the registered model.
    """
    raise NotImplementedError("ML stub — partner implements")


def load_model(model_name: str = "drift-triage-classifier") -> tuple[object, float]:
    """Load the Production model and its operating threshold from MLflow.

    Called in service/main.py lifespan to populate app.state.classifier
    and app.state.threshold.

    Args:
        model_name: MLflow registered model name.

    Returns:
        Tuple of (fitted sklearn Pipeline, operating threshold float).

    Raises:
        mlflow.exceptions.MlflowException: If no Production version exists.
    """
    import mlflow

    from core.settings import get_settings

    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    client = mlflow.MlflowClient()
    versions = client.get_latest_versions(model_name, stages=["Production"])
    if not versions:
        raise mlflow.exceptions.MlflowException(
            f"No Production model found for '{model_name}'. Run `make train` first."
        )
    version = versions[0]
    pipeline = mlflow.sklearn.load_model(version.source)
    threshold = float(
        client.get_model_version_tag(model_name, version.version, "threshold")
    )
    return pipeline, threshold
