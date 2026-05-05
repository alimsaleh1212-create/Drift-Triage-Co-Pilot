"""Database session dependency for the model service."""

from __future__ import annotations

from typing import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from core.settings import get_settings

_engine = create_async_engine(
    get_settings().async_database_url,
    pool_pre_ping=True,
)
_SessionLocal = async_sessionmaker(_engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_session() -> AsyncIterator[AsyncSession]:
    """Yield an async DB session scoped to the request."""
    async with _SessionLocal() as session:
        yield session
