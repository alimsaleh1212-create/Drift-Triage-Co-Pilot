"""Tool: enqueue approved action to arq worker queue with idempotency."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agent.tools.base import BaseTool
from worker.dedup import enqueue_with_dedup


class DispatchActionInput(BaseModel):
    investigation_id: str
    action: Literal["replay_test", "retrain", "rollback"]
    hil_approval_id: str
    model_version: int


class DispatchActionOutput(BaseModel):
    job_id: str
    idempotency_key: str
    status: Literal["enqueued", "deduplicated"]


class DispatchAction(BaseTool[DispatchActionInput, DispatchActionOutput]):
    """Enqueue an approved action to the arq worker queue."""

    name = "dispatch_action"
    input_schema = DispatchActionInput
    output_schema = DispatchActionOutput

    async def run(self, args: DispatchActionInput) -> DispatchActionOutput:
        idempotency_key = f"{args.action}:{args.investigation_id}"
        result = await enqueue_with_dedup(
            job_type=args.action,
            investigation_id=args.investigation_id,
            idempotency_key=idempotency_key,
            payload={
                "model_version": args.model_version,
                "hil_approval_id": args.hil_approval_id,
            },
        )
        return DispatchActionOutput(
            job_id=result["job_id"],
            idempotency_key=idempotency_key,
            status=result["status"],
        )
