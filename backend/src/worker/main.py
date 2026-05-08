"""arq worker entry point: imports job functions from dedicated modules."""

from __future__ import annotations

from typing import Any

import arq

from core.logging import configure_logging, get_logger
from core.settings import get_settings
from worker.replay_test import replay_test
from worker.retrain import retrain
from worker.rollback import rollback

log = get_logger(__name__)


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
