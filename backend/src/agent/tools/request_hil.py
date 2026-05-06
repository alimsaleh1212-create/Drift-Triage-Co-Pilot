"""Tool: record a HIL approval request in Postgres."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field
from sqlalchemy import text

from agent.deps.db import get_session
from agent.tools.base import BaseTool
from core.logging import get_logger

log = get_logger(__name__)


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
        approval_id = str(uuid4())
        async with get_session() as session:
            await session.execute(
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
            await session.commit()
        return RequestHILOutput(
            approval_id=approval_id,
            status="pending",
            created_at=datetime.now(timezone.utc),
        )
