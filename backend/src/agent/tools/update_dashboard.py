"""Tool: write investigation summary to Postgres for dashboard consumption."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel
from sqlalchemy import text

from agent.deps.db import get_session
from agent.tools.base import BaseTool


class UpdateDashboardInput(BaseModel):
    investigation_id: str
    summary_md: str
    status: Literal["open", "resolved", "awaiting_approval"]


class UpdateDashboardOutput(BaseModel):
    updated_at: datetime


class UpdateDashboard(BaseTool[UpdateDashboardInput, UpdateDashboardOutput]):
    """Persist investigation summary to the investigations table."""

    name = "update_dashboard"
    input_schema = UpdateDashboardInput
    output_schema = UpdateDashboardOutput

    async def run(self, args: UpdateDashboardInput) -> UpdateDashboardOutput:
        async with get_session() as session:
            await session.execute(
                text(
                    "UPDATE investigations "
                    "SET summary_md = :summary, status = :status, updated_at = now() "
                    "WHERE id = :id"
                ),
                {"summary": args.summary_md, "status": args.status, "id": args.investigation_id},
            )
            await session.commit()
        return UpdateDashboardOutput(updated_at=datetime.now(timezone.utc))
