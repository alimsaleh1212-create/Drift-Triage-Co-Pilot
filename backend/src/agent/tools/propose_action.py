"""Tool: propose a response action based on triage analysis."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agent.tools.base import BaseTool


class ProposeActionInput(BaseModel):
    investigation_id: str
    severity: Literal["low", "medium", "high"]
    drifted_features: list[str]
    hypothesis: str = Field(..., min_length=1, max_length=1024)


class ProposeActionOutput(BaseModel):
    action: Literal["replay_test", "retrain", "rollback", "no_action"]
    rationale: str
    requires_hil: bool
    priority: Literal["low", "medium", "high"]


class ProposeAction(BaseTool[ProposeActionInput, ProposeActionOutput]):
    """Propose a remediation action given triage analysis."""

    name = "propose_action"
    input_schema = ProposeActionInput
    output_schema = ProposeActionOutput

    async def run(self, args: ProposeActionInput) -> ProposeActionOutput:
        # Simple rule-based proposal; upgrade to LLM call if needed
        if args.severity == "low":
            return ProposeActionOutput(
                action="no_action",
                rationale="Severity too low to warrant intervention.",
                requires_hil=False,
                priority="low",
            )
        action: Literal["retrain", "replay_test"] = (
            "retrain" if args.severity == "high" else "replay_test"
        )
        return ProposeActionOutput(
            action=action,
            rationale=(
                f"Severity={args.severity}. "
                f"Features drifted: {args.drifted_features}."
            ),
            requires_hil=True,  # all Production-touching actions require HIL
            priority=args.severity,
        )
