"""Database session dependency for the model service.

Engine and session factory are created during lifespan, stored on
``app.state``, and injected via ``Depends()`` — no module-level globals.
"""

from __future__ import annotations

from typing import AsyncIterator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from ml.reference_stats import ReferenceStats


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Yield an async DB session scoped to the request.

    Uses the ``SessionLocal`` factory attached to ``app.state`` during
    lifespan startup.  FastAPI manages the session lifecycle.
    """
    session: AsyncSession = request.app.state.SessionLocal()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()