"""arq worker: replay_test, retrain, rollback jobs with DLQ on exhaustion."""

from __future__ import annotations

from typing import Any

import arq
from arq import Retry

from core.logging import configure_logging, get_logger
from core.settings import get_settings

log = get_logger(__name__)

_DLQ_KEY = "drift_actions:dlq"


async def _push_dlq(ctx: dict[str, Any], job_type: str, payload: dict[str, Any]) -> None:
    import json

    await ctx["redis"].rpush(
        _DLQ_KEY,
        json.dumps({"job_type": job_type, "payload": payload}),
    )
    log.error("worker.dlq", job_type=job_type)


async def replay_test(
    ctx: dict[str, Any],
    investigation_id: str,
    idempotency_key: str,
    payload: dict[str, Any],
) -> None:
    """Run the held-out test set through the current Production model and log metrics."""
    log.info("worker.replay_test.start", investigation_id=investigation_id)
    try:
        import mlflow
        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

        from core.settings import get_settings
        from ml.data import load_data
        from ml.inference import predict_batch
        from ml.register import load_model

        settings = get_settings()
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

        pipeline, threshold = load_model()
        split = load_data()
        proba = predict_batch(split.X_test, pipeline)
        labels = (proba >= threshold).astype(int)

        metrics = {
            "replay_accuracy": float(accuracy_score(split.y_test, labels)),
            "replay_f1": float(f1_score(split.y_test, labels)),
            "replay_precision": float(precision_score(split.y_test, labels, zero_division=0)),
            "replay_recall": float(recall_score(split.y_test, labels)),
            "replay_auc": float(roc_auc_score(split.y_test, proba)),
        }

        client = mlflow.MlflowClient()
        versions = client.get_latest_versions("bank-marketing-classifier", stages=["Production"])
        if versions:
            run_id = versions[0].run_id
            mlflow.start_run(run_id=run_id)
            mlflow.log_metrics(metrics)
            mlflow.end_run()

        log.info("worker.replay_test.done", investigation_id=investigation_id, **metrics)
    except Exception:
        log.exception("worker.replay_test.error", investigation_id=investigation_id)
        if ctx.get("job_try", 1) >= 3:
            await _push_dlq(ctx, "replay_test", payload)
            return
        raise Retry(defer=2 ** ctx.get("job_try", 1))


async def retrain(
    ctx: dict[str, Any],
    investigation_id: str,
    idempotency_key: str,
    payload: dict[str, Any],
) -> None:
    """Full retrain pipeline on current data, register as Staging."""
    log.info("worker.retrain.start", investigation_id=investigation_id)
    try:
        from ml.data import load_data
        from ml.reference_stats import compute_reference_stats
        from ml.register import register_model
        from ml.threshold import find_threshold
        from ml.train import train

        split = load_data()
        result = train(split)
        threshold = find_threshold(result.pipeline, split.X_val, split.y_val)
        ref_stats = compute_reference_stats(result.pipeline, split)
        run_id = register_model(result, threshold, ref_stats, split.dataset_hash)
        log.info("worker.retrain.done", investigation_id=investigation_id, run_id=run_id)
    except Exception:
        log.exception("worker.retrain.error", investigation_id=investigation_id)
        if ctx.get("job_try", 1) >= 3:
            await _push_dlq(ctx, "retrain", payload)
            return
        raise Retry(defer=2 ** ctx.get("job_try", 1))


async def rollback(
    ctx: dict[str, Any],
    investigation_id: str,
    idempotency_key: str,
    payload: dict[str, Any],
) -> None:
    """Re-promote a previous stable version to Production via the promotion gate."""
    log.info("worker.rollback.start", investigation_id=investigation_id)
    try:
        import httpx

        settings = get_settings()
        target_version = payload.get("model_version")
        hil_approval_id = payload.get("hil_approval_id", "")
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{settings.service_url}/api/v1/promotion/promote",
                headers={"X-Promotion-Key": settings.promotion_api_key},
                json={
                    "model_name": "drift-triage-classifier",
                    "target_version": target_version,
                    "investigation_id": investigation_id,
                    "hil_approval_id": hil_approval_id,
                },
            )
            r.raise_for_status()
        log.info("worker.rollback.done", investigation_id=investigation_id)
    except Exception:
        log.exception("worker.rollback.error", investigation_id=investigation_id)
        if ctx.get("job_try", 1) >= 3:
            await _push_dlq(ctx, "rollback", payload)
            return
        raise Retry(defer=2 ** ctx.get("job_try", 1))


async def startup(ctx: dict[str, Any]) -> None:
    configure_logging()
    log.info("worker.startup")


async def shutdown(ctx: dict[str, Any]) -> None:
    log.info("worker.shutdown")


class WorkerSettings:
    """arq worker configuration."""

    redis_settings = arq.connections.RedisSettings.from_dsn(get_settings().redis_url)
    functions = [replay_test, retrain, rollback]
    on_startup = startup
    on_shutdown = shutdown
    max_jobs = 10
    job_timeout = 3600  # 1 hour max per job
    keep_result = 3600
    queue_name = "drift_actions"
    retry_jobs = True
    max_tries = 3
