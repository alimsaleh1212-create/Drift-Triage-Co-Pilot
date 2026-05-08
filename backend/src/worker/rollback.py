"""arq worker job: re-promote previous stable Production version via promotion gate."""

from __future__ import annotations

from typing import Any

from arq import Retry

from core.logging import get_logger
from core.settings import get_settings
from worker.dedup import push_to_dlq

log = get_logger(__name__)


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

        from ml.register import MODEL_NAME

        settings = get_settings()
        target_version = payload.get("model_version")
        hil_approval_id = payload.get("hil_approval_id", "")
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{settings.service_url}/api/v1/promotion/promote",
                headers={"X-Promotion-Key": settings.promotion_api_key},
                json={
                    "model_name": MODEL_NAME,
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
            await push_to_dlq(ctx, "rollback", payload)
            return
        raise Retry(defer=2 ** ctx.get("job_try", 1))
