"""arq worker job: replay test set through Production model and log metrics."""

from __future__ import annotations

from typing import Any

from arq import Retry

from core.logging import get_logger
from worker.dedup import push_to_dlq

log = get_logger(__name__)


async def replay_test(
    ctx: dict[str, Any],
    investigation_id: str,
    idempotency_key: str,
    payload: dict[str, Any],
) -> None:
    """Run held-out test set through Production model and log metrics."""
    log.info("worker.replay_test.start", investigation_id=investigation_id)
    try:
        import mlflow
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        from core.settings import get_settings
        from ml.data import load_data
        from ml.inference import predict_batch
        from ml.register import MODEL_NAME, load_model

        settings = get_settings()
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

        pipeline, threshold = load_model()
        split = load_data()
        proba = predict_batch(split.X_test, pipeline)
        labels = (proba >= threshold).astype(int)

        metrics = {
            "replay_accuracy": float(accuracy_score(split.y_test, labels)),
            "replay_f1": float(f1_score(split.y_test, labels)),
            "replay_precision": float(
                precision_score(split.y_test, labels, zero_division=0)
            ),
            "replay_recall": float(recall_score(split.y_test, labels)),
            "replay_auc": float(roc_auc_score(split.y_test, proba)),
        }

        client = mlflow.MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if versions:
            run_id = versions[0].run_id
            mlflow.start_run(run_id=run_id)
            mlflow.log_metrics(metrics)
            mlflow.end_run()

        log.info(
            "worker.replay_test.done", investigation_id=investigation_id, **metrics
        )
    except Exception:
        log.exception("worker.replay_test.error", investigation_id=investigation_id)
        if ctx.get("job_try", 1) >= 3:
            await push_to_dlq(ctx, "replay_test", payload)
            return
        raise Retry(defer=2 ** ctx.get("job_try", 1))
