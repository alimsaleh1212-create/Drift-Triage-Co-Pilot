"""Database session management for the agent service.

Engine and session factory are initialised once during lifespan, disposed on
shutdown.  All tools and endpoints use ``get_session()`` instead of creating
their own engines — per-call ``create_async_engine()`` is a connection leak.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from core.settings import get_settings

_engine: AsyncEngine | None = None
_SessionLocal: async_sessionmaker[AsyncSession] | None = None


def init_db() -> None:
    """Create engine and session factory.  Called once in agent lifespan."""
    global _engine, _SessionLocal
    settings = get_settings()
    _engine = create_async_engine(settings.async_database_url, pool_pre_ping=True)
    _SessionLocal = async_sessionmaker(_engine, expire_on_commit=False)


async def close_db() -> None:
    """Dispose engine.  Called once in agent lifespan shutdown."""
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """Yield an async DB session scoped to the caller.

    Tools use this instead of creating their own engine per call.
    """
    if _SessionLocal is None:
        raise RuntimeError("Database not initialised. Call init_db() first.")
    async with _SessionLocal() as session:
        yield session