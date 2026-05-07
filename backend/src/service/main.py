"""FastAPI model service: prediction, drift reporting, and promotion gate."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from core.logging import configure_logging, get_logger
from core.settings import get_settings
from ml.reference_stats import load_reference_stats
from ml.register import MODEL_NAME, load_model
from service.routers import drift, prediction, promotion

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load singletons on startup; dispose on shutdown."""
    configure_logging()
    settings = get_settings()

    # Database — engine and session factory created once per process
    engine = create_async_engine(settings.async_database_url, pool_pre_ping=True)
    SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
    app.state.engine = engine
    app.state.SessionLocal = SessionLocal

    # ML artefacts — loaded once, reused per request
    pipeline, threshold = load_model()
    ref_stats = load_reference_stats()

    app.state.classifier = pipeline
    app.state.threshold = threshold
    app.state.ref_stats = ref_stats
    app.state.model_name = MODEL_NAME

    # Shared async HTTP client for webhook delivery
    app.state.http_client = httpx.AsyncClient(timeout=10.0)

    log.info("service.startup", model_name=app.state.model_name, threshold=threshold)
    yield

    await app.state.http_client.aclose()
    await engine.dispose()
    log.info("service.shutdown")


app = FastAPI(
    title="Drift Triage — Model Service",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(prediction.router, prefix="/api/v1", tags=["prediction"])
app.include_router(drift.router, prefix="/api/v1", tags=["drift"])
app.include_router(promotion.router, prefix="/api/v1", tags=["promotion"])


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
