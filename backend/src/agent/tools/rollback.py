"""Tool: enqueue a rollback to the previous stable Production version."""

from __future__ import annotations

from pydantic import BaseModel

from agent.tools.base import BaseTool


class RollbackInput(BaseModel):
    investigation_id: str
    hil_approval_id: str
    target_version: int


class RollbackOutput(BaseModel):
    job_id: str
    status: str


class Rollback(BaseTool[RollbackInput, RollbackOutput]):
    """Re-promote the previous stable Production version."""

    name = "rollback"
    input_schema = RollbackInput
    output_schema = RollbackOutput

    async def run(self, args: RollbackInput) -> RollbackOutput:
        from worker.dedup import enqueue_with_dedup

        result = await enqueue_with_dedup(
            job_type="rollback",
            investigation_id=args.investigation_id,
            idempotency_key=f"rollback:{args.investigation_id}",
            payload={
                "target_version": args.target_version,
                "hil_approval_id": args.hil_approval_id,
            },
        )
        return RollbackOutput(job_id=result["job_id"], status=result["status"])
