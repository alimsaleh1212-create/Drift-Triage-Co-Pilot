"""Tool: enqueue a replay-test job (deferred via arq)."""

from __future__ import annotations

from pydantic import BaseModel

from agent.tools.base import BaseTool


class ReplayTestInput(BaseModel):
    investigation_id: str
    model_version: int


class ReplayTestOutput(BaseModel):
    job_id: str
    status: str


class ReplayTest(BaseTool[ReplayTestInput, ReplayTestOutput]):
    """Enqueue a replay test against the held-out test set."""

    name = "replay_test"
    input_schema = ReplayTestInput
    output_schema = ReplayTestOutput

    async def run(self, args: ReplayTestInput) -> ReplayTestOutput:
        from worker.dedup import enqueue_with_dedup

        result = await enqueue_with_dedup(
            job_type="replay_test",
            investigation_id=args.investigation_id,
            idempotency_key=f"replay_test:{args.investigation_id}",
            payload={"model_version": args.model_version},
        )
        return ReplayTestOutput(job_id=result["job_id"], status=result["status"])
