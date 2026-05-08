"""GET /models/registry — live MLflow model registry summary."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter

from core.logging import get_logger
from core.settings import get_settings

router = APIRouter()
log = get_logger(__name__)


def _version_info(versions: list) -> dict[str, Any] | None:
    if not versions:
        return None
    v = versions[0]
    return {
        "version": int(v.version),
        "run_id": v.run_id,
        "auc": float(v.tags.get("auc") or 0),
        "recall": float(v.tags.get("recall") or 0),
        "threshold": float(v.tags.get("operating_threshold") or 0),
        "stage": v.current_stage,
    }


def _fetch_registry() -> dict[str, Any]:
    import mlflow

    from ml.register import MODEL_NAME

    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.MlflowClient()
    return {
        "model_name": MODEL_NAME,
        "production": _version_info(
            client.get_latest_versions(MODEL_NAME, stages=["Production"])
        ),
        "staging": _version_info(
            client.get_latest_versions(MODEL_NAME, stages=["Staging"])
        ),
    }


@router.get("/models/registry")
async def get_registry() -> dict[str, Any]:
    """Return live Production and Staging model version info from MLflow."""
    return await asyncio.to_thread(_fetch_registry)
