"""Redis SETNX-based idempotency guard for arq job dispatch.

Guarantees: one job per idempotency_key, even if enqueued twice.
Key pattern: dispatch:{idempotency_key}  TTL = 24h.
"""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

import redis.asyncio as aioredis

from core.logging import get_logger
from core.settings import get_settings

log = get_logger(__name__)

_DEDUP_TTL_SECONDS = 86_400  # 24 hours — covers max job runtime + buffer


async def enqueue_with_dedup(
    job_type: str,
    investigation_id: str,
    idempotency_key: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Enqueue a job to arq only if this idempotency_key has not been seen.

    Uses SETNX on Redis key ``dispatch:{idempotency_key}`` to guarantee
    at-most-once dispatch for the TTL window.

    Args:
        job_type: One of "replay_test", "retrain", "rollback".
        investigation_id: UUID of the owning investigation.
        idempotency_key: Unique key like "retrain:inv_abc123".
        payload: Extra args forwarded to the worker function.

    Returns:
        Dict with "job_id" and "status" ("enqueued" | "deduplicated").
    """
    settings = get_settings()
    redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    dedup_key = f"dispatch:{idempotency_key}"

    try:
        acquired = await redis.set(dedup_key, "1", nx=True, ex=_DEDUP_TTL_SECONDS)
        if not acquired:
            log.info("dedup.skipped", idempotency_key=idempotency_key)
            return {"job_id": idempotency_key, "status": "deduplicated"}

        import arq

        pool = await arq.create_pool(arq.connections.RedisSettings.from_dsn(settings.redis_url))
        job_id = str(uuid4())
        await pool.enqueue_job(
            job_type,
            investigation_id=investigation_id,
            idempotency_key=idempotency_key,
            payload=payload,
            _job_id=job_id,
        )
        await pool.close()
        log.info("dedup.enqueued", idempotency_key=idempotency_key, job_id=job_id)
        return {"job_id": job_id, "status": "enqueued"}
    finally:
        await redis.aclose()
