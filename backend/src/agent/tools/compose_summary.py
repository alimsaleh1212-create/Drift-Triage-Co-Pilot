"""Tool: compose investigation summary for dashboard and HIL approval request."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent.llm import call_llm
from agent.security import delimit_external, sanitize_feature_string
from agent.tools.base import BaseTool
from drift.severity import DriftReport


class ComposeSummaryInput(BaseModel):
    investigation_id: str
    report: DriftReport
    proposed_action: str
    triage_notes: str = Field(default="")


class ComposeSummaryOutput(BaseModel):
    summary_md: str
    hil_message: str


class _SummarySchema(BaseModel):
    summary_md: str
    hil_message: str


class ComposeSummary(BaseTool[ComposeSummaryInput, ComposeSummaryOutput]):
    """Use LLM to compose investigation summary and HIL message."""

    name = "compose_summary"
    input_schema = ComposeSummaryInput
    output_schema = ComposeSummaryOutput

    async def run(self, args: ComposeSummaryInput) -> ComposeSummaryOutput:
        import importlib.resources

        system_prompt = (
            importlib.resources.files("agent.prompts")
            .joinpath("comms.md")
            .read_text()
        )
        triage_notes = sanitize_feature_string(args.triage_notes)
        user_prompt = (
            f"{system_prompt}\n\n"
            f"Investigation ID: {args.investigation_id}\n"
            f"Severity: {args.report.severity}\n"
            f"Proposed action: {delimit_external(args.proposed_action)}\n"
            f"Triage notes: {delimit_external(triage_notes)}\n"
            "Respond with JSON matching the schema."
        )
        result = await call_llm(user_prompt, _SummarySchema)
        return ComposeSummaryOutput(
            summary_md=sanitize_feature_string(result.summary_md),
            hil_message=sanitize_feature_string(result.hil_message),
        )
