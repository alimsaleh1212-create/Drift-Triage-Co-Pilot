"""Agent snapshot trajectory tests — run without an API key.

FakeLLM replays recorded tool calls; asserts agent state at each step.
Any routing change → snapshot mismatch → update the JSON and explain in PR.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from tests.agent.conftest import load_snapshot


@pytest.mark.asyncio
async def test_high_severity_triage_proposes_retrain(
    fake_llm_factory: Any,
) -> None:
    """High-severity webhook → triage → action proposes retrain → HIL pause."""
    snapshot = load_snapshot("high_severity_retrain_path")
    webhook_data = snapshot["webhook"]
    expected = snapshot["expected_state_after"]

    fake_llm = fake_llm_factory("high_severity_retrain_path")

    with (
        patch("agent.llm.call_llm", side_effect=fake_llm),
        patch(
            "agent.tools.inspect_drift._fetch_report",
            new=AsyncMock(return_value=_make_drift_report(webhook_data)),
        ),
        patch(
            "agent.tools.request_hil.RequestHIL.run",
            new=AsyncMock(
                return_value=_make_hil_output("hil-test-001"),
            ),
        ),
        patch(
            "agent.tools.update_dashboard.UpdateDashboard.run",
            new=AsyncMock(return_value=_make_update_output()),
        ),
    ):
        from agent.graph import (
            AgentState,
            action_node,
            comms_node,
            triage_node,
        )

        state: AgentState = {
            "investigation_id": "test-inv",
            "alert": webhook_data,
            "report": None,
            "triage_notes": "",
            "proposed_action": None,
            "requires_hil": False,
            "hil_approval_id": None,
            "awaiting_approval": False,
            "summary_md": "",
            "dispatch_status": "",
            "status": "open",
            "drift_report_id": webhook_data["report_id"],
        }

        # Step 1 — triage
        state = await triage_node(state)
        exp_triage = expected[0]["fields"]
        assert state["alert"]["severity"] == exp_triage["severity_in_alert"]
        assert bool(state["triage_notes"]) == exp_triage["triage_notes_nonempty"]

        # Step 2 — action
        with patch(
            "agent.staleness.assert_not_stale",
            new=AsyncMock(return_value=None),
        ):
            state = await action_node(state)
        exp_action = expected[1]["fields"]
        assert state["proposed_action"] == exp_action["proposed_action"]
        assert state["awaiting_approval"] == exp_action["awaiting_approval"]

        # Step 3 — comms
        state = await comms_node(state)
        exp_comms = expected[2]["fields"]
        assert bool(state["summary_md"]) == exp_comms["summary_md_nonempty"]


def _make_drift_report(webhook_data: dict[str, Any]) -> Any:
    from drift.severity import DriftReport
    return DriftReport.model_validate(webhook_data)


def _make_hil_output(approval_id: str) -> Any:
    from datetime import datetime, timezone
    from agent.tools.request_hil import RequestHILOutput
    return RequestHILOutput(
        approval_id=approval_id,
        status="pending",
        created_at=datetime.now(timezone.utc),
    )


def _make_update_output() -> Any:
    from datetime import datetime, timezone
    from agent.tools.update_dashboard import UpdateDashboardOutput
    return UpdateDashboardOutput(updated_at=datetime.now(timezone.utc))
