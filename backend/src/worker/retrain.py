"""arq worker job: full retrain on current data; register as Staging."""

from __future__ import annotations

from typing import Any

from arq import Retry

from core.logging import get_logger
from core.settings import get_settings
from worker.dedup import push_to_dlq

log = get_logger(__name__)


async def retrain(
    ctx: dict[str, Any],
    investigation_id: str,
    idempotency_key: str,
    payload: dict[str, Any],
) -> None:
    """Full retrain on current data; register as Staging, auto-promote if better."""
    log.info("worker.retrain.start", investigation_id=investigation_id)
    try:
        import mlflow

        from ml.data import load_data
        from ml.reference_stats import compute_reference_stats
        from ml.register import MODEL_NAME, register_model
        from ml.threshold import find_threshold
        from ml.train import train

        settings = get_settings()
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client = mlflow.MlflowClient()

        split = load_data()
        result = train(split)
        threshold = find_threshold(result.pipeline, split.X_val, split.y_val)
        ref_stats = compute_reference_stats(result.pipeline, split)
        run_id = register_model(result, threshold, ref_stats, split.dataset_hash)
        log.info(
            "worker.retrain.done",
            investigation_id=investigation_id,
            run_id=run_id,
            auc=result.auc,
        )

        staging_versions = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
        if not staging_versions:
            log.warning("worker.retrain.no_staging", investigation_id=investigation_id)
            return

        new_version = int(staging_versions[0].version)
        new_auc = result.auc

        prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        prod_auc = float(prod_versions[0].tags.get("auc", 0)) if prod_versions else 0.0

        if new_auc >= prod_auc:
            import httpx

            hil_approval_id = payload.get("hil_approval_id", "")
            async with httpx.AsyncClient(timeout=30.0) as http:
                r = await http.post(
                    f"{settings.service_url}/api/v1/promotion/promote",
                    headers={"X-Promotion-Key": settings.promotion_api_key},
                    json={
                        "model_name": MODEL_NAME,
                        "target_version": new_version,
                        "investigation_id": investigation_id,
                        "hil_approval_id": hil_approval_id,
                    },
                )
                if not r.is_success:
                    log.warning(
                        "worker.retrain.promotion_failed",
                        investigation_id=investigation_id,
                        new_version=new_version,
                        status=r.status_code,
                        detail=r.text,
                    )
                r.raise_for_status()
            log.info(
                "worker.retrain.promoted",
                investigation_id=investigation_id,
                new_version=new_version,
                new_auc=new_auc,
                prod_auc=prod_auc,
            )
        else:
            log.info(
                "worker.retrain.kept_production",
                investigation_id=investigation_id,
                new_auc=new_auc,
                prod_auc=prod_auc,
            )
    except Exception:
        log.exception("worker.retrain.error", investigation_id=investigation_id)
        if ctx.get("job_try", 1) >= 3:
            await push_to_dlq(ctx, "retrain", payload)
            return
        raise Retry(defer=2 ** ctx.get("job_try", 1))
