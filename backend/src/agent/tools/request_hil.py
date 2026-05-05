"""Tool: record a HIL approval request in Postgres."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from agent.tools.base import BaseTool


class RequestHILInput(BaseModel):
    investigation_id: str
    action: Literal["replay_test", "retrain", "rollback"]
    rationale: str = Field(..., min_length=1, max_length=2048)
    model_version: int


class RequestHILOutput(BaseModel):
    approval_id: str
    status: Literal["pending"]
    created_at: datetime


class RequestHIL(BaseTool[RequestHILInput, RequestHILOutput]):
    """Create a HIL approval request; agent state pauses until approval."""

    name = "request_hil_approval"
    input_schema = RequestHILInput
    output_schema = RequestHILOutput

    async def run(self, args: RequestHILInput) -> RequestHILOutput:
        # Persist to Postgres hil_approvals table
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine

        from core.settings import get_settings

        settings = get_settings()
        engine = create_async_engine(settings.async_database_url)
        approval_id = str(uuid4())
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO hil_approvals "
                    "(id, investigation_id, action, rationale, model_version, status, created_at) "
                    "VALUES (:id, :inv_id, :action, :rationale, :mv, 'pending', now())"
                ),
                {
                    "id": approval_id,
                    "inv_id": args.investigation_id,
                    "action": args.action,
                    "rationale": args.rationale,
                    "mv": args.model_version,
                },
            )
        await engine.dispose()
        return RequestHILOutput(
            approval_id=approval_id,
            status="pending",
            created_at=datetime.now(timezone.utc),
        )
