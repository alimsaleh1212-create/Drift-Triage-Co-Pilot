"""FastAPI agent service: webhook intake, HIL approval, investigations API."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from agent.deps.db import close_db, init_db
from agent.graph import build_graph
from agent.routers import admin, hil, investigations, queue, webhook
from core.logging import configure_logging, get_logger
from core.settings import Settings, get_settings

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    init_db()
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    settings = get_settings()
    _configure_langsmith(settings)
    async with AsyncPostgresSaver.from_conn_string(
        settings.checkpoint_database_url
    ) as checkpointer:
        await checkpointer.setup()
        app.state.graph = build_graph(checkpointer=checkpointer)
        log.info("agent.startup")
        yield
    await close_db()
    log.info("agent.shutdown")


app = FastAPI(title="Drift Triage — Agent", version="0.1.0", lifespan=lifespan)

app.include_router(webhook.router, tags=["webhook"])
app.include_router(hil.router, tags=["hil"])
app.include_router(investigations.router, tags=["investigations"])
app.include_router(queue.router, tags=["queue"])
app.include_router(admin.router, tags=["admin"])


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


def _configure_langsmith(settings: Settings) -> None:
    """Expose Settings-backed LangSmith values to LangGraph's env-based tracer."""
    if not settings.langsmith_tracing:
        return

    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
    if settings.langsmith_api_key:
        os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)

    log.info(
        "langsmith.tracing_configured",
        project=settings.langsmith_project,
    )
