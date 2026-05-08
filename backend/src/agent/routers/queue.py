"""Queue metrics endpoint: GET /queue/metrics."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from core.logging import get_logger
from core.settings import get_settings

router = APIRouter()
log = get_logger(__name__)


@router.get("/queue/metrics")
async def queue_metrics() -> dict[str, Any]:
    """Return arq queue depth, DLQ size, dedup locks, and recent worker jobs."""
    import json

    import redis.asyncio as aioredis

    settings = get_settings()
    redis_str = aioredis.from_url(settings.redis_url, decode_responses=True)
    redis_bytes = aioredis.from_url(settings.redis_url, decode_responses=False)
    try:
        queue_key = f"arq:queue:{settings.redis_queue_name}"
        dlq_key = f"{settings.redis_queue_name}:dlq"
        queue_depth = await redis_str.zcard(queue_key)
        dlq_count = await redis_str.llen(dlq_key)
        active_dispatches = 0
        async for _ in redis_str.scan_iter(match="dispatch:*", count=100):
            active_dispatches += 1
        result_keys = []
        async for key in redis_bytes.scan_iter(match="arq:result:*", count=200):
            result_keys.append(key)
        recent_jobs: list[dict[str, Any]] = []
        for key in result_keys[-20:]:
            raw = await redis_bytes.get(key)
            if not raw:
                continue
            key_str = key.decode("utf-8") if isinstance(key, bytes) else key
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = {"raw": f"binary ({len(raw)} bytes)"}
            recent_jobs.append({"key": key_str, **parsed})
        dlq_items: list[dict[str, Any]] = []
        for raw in await redis_str.lrange(dlq_key, 0, 9):
            try:
                dlq_items.append(json.loads(raw))
            except Exception:
                dlq_items.append({"raw": raw})
        worker_running = bool(active_dispatches > 0 and queue_depth == 0)
        return {
            "queue_depth": int(queue_depth or 0),
            "dlq_count": int(dlq_count or 0),
            "active_dispatches": active_dispatches,
            "worker_running": worker_running,
            "recent_jobs_count": len(recent_jobs),
            "recent_jobs": recent_jobs,
            "dlq_items": dlq_items,
        }
    finally:
        await redis_str.aclose()
        await redis_bytes.aclose()
