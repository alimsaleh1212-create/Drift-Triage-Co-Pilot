"""Investigation endpoints: GET /investigations, GET /investigations/{id}."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from agent.deps.db import get_session

router = APIRouter()


@router.get("/investigations")
async def list_investigations() -> list[dict[str, Any]]:
    """List recent investigations from Postgres."""
    from sqlalchemy import text

    async with get_session() as session:
        result = await session.execute(
            text(
                "SELECT "
                "id, drift_event_id, drift_report_id, status, summary_md, updated_at "
                "FROM investigations "
                "ORDER BY updated_at DESC "
                "LIMIT 50"
            )
        )
        return [dict(r._mapping) for r in result]


@router.get("/investigations/{investigation_id}")
async def get_investigation(
    investigation_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Return a single investigation by ID."""
    from sqlalchemy import text

    result = await session.execute(
        text("SELECT * FROM investigations WHERE id = :id"),
        {"id": investigation_id},
    )
    row = result.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Investigation not found")
    return dict(row._mapping)
