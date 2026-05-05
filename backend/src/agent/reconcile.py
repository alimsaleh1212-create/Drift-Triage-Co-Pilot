"""Checkpoint reconciliation: handle model-URI-gone on wakeup.

Per CLAUDE.md §16 think-about questions:
Q: What if the model URI is gone when the agent wakes from a checkpoint?
A: reconcile.py checks that the MLflow model URI referenced in the checkpoint
   still resolves. If not, it logs a warning and re-fetches the current
   Production model URI before resuming.
"""

from __future__ import annotations

from core.logging import get_logger

log = get_logger(__name__)


async def reconcile_model_uri(model_name: str, stored_version: int) -> int:
    """Verify stored model version still exists; return a valid version.

    Called at agent wakeup from a Postgres checkpoint before any tool
    that references the model version.

    Args:
        model_name: MLflow registered model name.
        stored_version: Version integer stored in the checkpoint state.

    Returns:
        The stored version if it still exists, else the current Production version.
    """
    import mlflow

    from core.settings import get_settings

    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.MlflowClient()

    try:
        client.get_model_version(model_name, str(stored_version))
        log.info("reconcile.ok", model_name=model_name, version=stored_version)
        return stored_version
    except Exception:
        log.warning(
            "reconcile.version_gone",
            model_name=model_name,
            stored_version=stored_version,
        )
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        if not prod_versions:
            raise RuntimeError(
                f"No Production version for {model_name!r} — cannot reconcile."
            )
        current = int(prod_versions[0].version)
        log.info("reconcile.fallback", model_name=model_name, fallback_version=current)
        return current
