"""LangGraph supervisor graph: triage → action → (HIL) → comms → dispatch.

Topology (per CLAUDE.md §16):
    webhook_received → triage → should_act?
        ├─ yes → action → needs_approval?
        │         ├─ yes → pause_for_human → (resume) → comms → dispatch → END
        │         └─ no  → comms → dispatch → END
        └─ no  → comms → END

Postgres checkpoint persists state across restarts.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from agent.tools.compose_summary import ComposeSummary
from agent.tools.dispatch_action import DispatchAction
from agent.tools.inspect_drift import InspectDrift
from agent.tools.propose_action import ProposeAction
from agent.tools.request_hil import RequestHIL
from agent.tools.update_dashboard import UpdateDashboard
from core.logging import get_logger
from drift.severity import DriftReport

log = get_logger(__name__)

# Tool registry — allowlist enforced here; unknown tool names are rejected
TOOL_REGISTRY: dict[str, Any] = {
    t.name: t
    for t in [
        InspectDrift(),
        ProposeAction(),
        RequestHIL(),
        DispatchAction(),
        ComposeSummary(),
        UpdateDashboard(),
    ]
}


class AgentState(TypedDict):
    """LangGraph state persisted to Postgres checkpointer."""

    investigation_id: str
    alert: dict[str, Any]  # DriftWebhookPayload serialized
    report: dict[str, Any] | None
    triage_notes: str
    proposed_action: str | None
    requires_hil: bool
    hil_approval_id: str | None
    awaiting_approval: bool
    summary_md: str
    dispatch_status: str
    status: Literal["open", "awaiting_approval", "resolved"]
    drift_report_id: str  # for staleness guard


async def triage_node(state: AgentState) -> AgentState:
    """Fetch drift report and run triage analysis."""
    from agent.llm import call_llm
    from agent.security import delimit_external
    from pydantic import BaseModel

    class TriageOutput(BaseModel):
        drifted_features: list[str]
        severity: str
        hypothesis: str
        should_act: bool

    tool = TOOL_REGISTRY["inspect_drift"]
    result = await tool.safe_run({"model_name": state["alert"].get("model_name", "")})
    if not result.ok:
        log.error("triage.inspect_drift_failed", error=result.error)
        return {**state, "triage_notes": f"inspect_drift failed: {result.error}"}

    report: DriftReport = result.result.report  # type: ignore[union-attr]
    report_dict = report.model_dump(mode="json")

    import importlib.resources

    system_prompt = (
        importlib.resources.files("agent.prompts")
        .joinpath("triage.md")
        .read_text()
    )
    user_prompt = (
        f"{system_prompt}\n\n"
        f"Drift report:\n{delimit_external(str(report_dict))}\n"
        "Respond with JSON matching the schema."
    )
    analysis = await call_llm(user_prompt, TriageOutput)

    return {
        **state,
        "report": report_dict,
        "triage_notes": analysis.hypothesis,
        "drift_report_id": report.report_id,
    }


async def action_node(state: AgentState) -> AgentState:
    """Propose action and optionally request HIL approval."""
    from agent.staleness import assert_not_stale
    from drift.severity import DriftReport

    # Staleness guard before proposing any action
    if state.get("drift_report_id") and state.get("report"):
        current_report = DriftReport.model_validate(state["report"])
        try:
            await assert_not_stale(state["drift_report_id"], current_report)
        except Exception as exc:
            log.warning("action.stale", error=str(exc))
            return {**state, "status": "resolved", "summary_md": str(exc)}

    propose_tool = TOOL_REGISTRY["propose_action"]
    proposal_result = await propose_tool.safe_run(
        {
            "investigation_id": state["investigation_id"],
            "severity": state["alert"].get("severity", "low"),
            "drifted_features": [],
            "hypothesis": state.get("triage_notes", ""),
        }
    )
    if not proposal_result.ok:
        return {**state, "proposed_action": None, "requires_hil": False}

    proposal = proposal_result.result  # type: ignore[union-attr]

    if proposal.requires_hil:
        hil_tool = TOOL_REGISTRY["request_hil_approval"]
        hil_result = await hil_tool.safe_run(
            {
                "investigation_id": state["investigation_id"],
                "action": proposal.action,
                "rationale": proposal.rationale,
                "model_version": state["alert"].get("model_version", 1),
            }
        )
        if hil_result.ok:
            return {
                **state,
                "proposed_action": proposal.action,
                "requires_hil": True,
                "awaiting_approval": True,
                "hil_approval_id": hil_result.result.approval_id,  # type: ignore[union-attr]
                "status": "awaiting_approval",
            }

    return {
        **state,
        "proposed_action": proposal.action,
        "requires_hil": False,
        "awaiting_approval": False,
    }


async def comms_node(state: AgentState) -> AgentState:
    """Compose and persist investigation summary."""
    from drift.severity import DriftReport

    if not state.get("report"):
        return state

    compose_tool = TOOL_REGISTRY["compose_summary"]
    result = await compose_tool.safe_run(
        {
            "investigation_id": state["investigation_id"],
            "report": state["report"],
            "proposed_action": state.get("proposed_action") or "no_action",
            "triage_notes": state.get("triage_notes") or "",
        }
    )
    summary_md = result.result.summary_md if result.ok else "Summary generation failed."  # type: ignore[union-attr]

    update_tool = TOOL_REGISTRY["update_dashboard"]
    await update_tool.safe_run(
        {
            "investigation_id": state["investigation_id"],
            "summary_md": summary_md,
            "status": state.get("status", "open"),
        }
    )
    return {**state, "summary_md": summary_md}


async def dispatch_node(state: AgentState) -> AgentState:
    """Enqueue approved action to the arq worker."""
    if state.get("awaiting_approval") or not state.get("proposed_action"):
        return state

    dispatch_tool = TOOL_REGISTRY["dispatch_action"]
    result = await dispatch_tool.safe_run(
        {
            "investigation_id": state["investigation_id"],
            "action": state["proposed_action"],
            "hil_approval_id": state.get("hil_approval_id") or "",
            "model_version": state["alert"].get("model_version", 1),
        }
    )
    status = result.result.status if result.ok else "error"  # type: ignore[union-attr]
    return {**state, "dispatch_status": status, "status": "resolved"}


def _should_act(state: AgentState) -> Literal["action", "comms"]:
    severity = state["alert"].get("severity", "low")
    return "action" if severity in ("medium", "high") else "comms"


def _needs_approval(state: AgentState) -> Literal["pause_for_human", "comms"]:
    return "pause_for_human" if state.get("awaiting_approval") else "comms"


def build_graph() -> Any:
    """Construct the LangGraph supervisor graph with Postgres checkpointer."""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    from core.settings import get_settings

    settings = get_settings()
    checkpointer = AsyncPostgresSaver.from_conn_string(settings.sync_database_url)
    checkpointer.setup()

    graph = StateGraph(AgentState)
    graph.add_node("triage", triage_node)
    graph.add_node("action", action_node)
    graph.add_node("pause_for_human", lambda s: s)  # interrupt point
    graph.add_node("comms", comms_node)
    graph.add_node("dispatch", dispatch_node)

    graph.set_entry_point("triage")
    graph.add_conditional_edges("triage", _should_act, {"action": "action", "comms": "comms"})
    graph.add_conditional_edges(
        "action",
        _needs_approval,
        {"pause_for_human": "pause_for_human", "comms": "comms"},
    )
    graph.add_edge("pause_for_human", END)  # resumes externally via HIL approve
    graph.add_edge("comms", "dispatch")
    graph.add_edge("dispatch", END)

    return graph.compile(checkpointer=checkpointer, interrupt_before=["pause_for_human"])
