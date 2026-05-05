"""Tool: enqueue a full retrain job (deferred via arq)."""

from __future__ import annotations

from pydantic import BaseModel

from agent.tools.base import BaseTool


class RetrainInput(BaseModel):
    investigation_id: str
    hil_approval_id: str


class RetrainOutput(BaseModel):
    job_id: str
    status: str


class Retrain(BaseTool[RetrainInput, RetrainOutput]):
    """Enqueue a full retrain on current data, registers as Staging."""

    name = "retrain"
    input_schema = RetrainInput
    output_schema = RetrainOutput

    async def run(self, args: RetrainInput) -> RetrainOutput:
        from worker.dedup import enqueue_with_dedup

        result = await enqueue_with_dedup(
            job_type="retrain",
            investigation_id=args.investigation_id,
            idempotency_key=f"retrain:{args.investigation_id}",
            payload={"hil_approval_id": args.hil_approval_id},
        )
        return RetrainOutput(job_id=result["job_id"], status=result["status"])
