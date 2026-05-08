"""Admin endpoints: POST /admin/reset."""

from __future__ import annotations

from fastapi import APIRouter

from agent.deps.db import get_session
from core.logging import get_logger

router = APIRouter()
log = get_logger(__name__)


@router.post("/admin/reset")
async def admin_reset() -> dict[str, str]:
    """Mark all open investigations resolved and clear pending HIL approvals."""
    from sqlalchemy import text

    async with get_session() as session:
        await session.execute(
            text(
                "UPDATE investigations SET status = 'resolved',"
                " updated_at = now() WHERE status != 'resolved'"
            )
        )
        await session.execute(
            text(
                "UPDATE hil_approvals SET status = 'rejected',"
                " decision = 'rejected' WHERE status = 'pending'"
            )
        )
        await session.commit()
    log.info("admin.reset")
    return {"status": "reset"}
