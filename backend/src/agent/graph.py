"""LangGraph supervisor graph for the Drift Triage Co-Pilot.

Topology:
    webhook_received → supervisor → triage_agent → supervisor
        → action_agent → supervisor
        → request_hil → pause_for_human → supervisor
        → dispatch → comms_agent → supervisor → END

The supervisor is the only router. Sub-agent nodes do bounded work and return
control to the supervisor, while Postgres checkpointing persists state across
HIL interrupts and service restarts.
"""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict

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


NextStep = Literal[
    "triage_agent",
    "action_agent",
    "request_hil",
    "pause_for_human",
    "dispatch",
    "comms_agent",
    "end",
]


class AgentState(TypedDict):
    """LangGraph state persisted to Postgres checkpointer."""

    investigation_id: str
    alert: dict[str, Any]  # DriftWebhookPayload serialized
    report: dict[str, Any] | None
    triage_notes: str
    proposed_action: str | None
    action_rationale: NotRequired[str]
    requires_hil: bool
    hil_approval_id: str | None
    awaiting_approval: bool
    summary_md: str
    dispatch_status: str
    status: Literal["open", "awaiting_approval", "resolved"]
    drift_report_id: str  # for staleness guard
    next_step: NotRequired[NextStep]
    comms_done: NotRequired[bool]


async def supervisor_node(state: AgentState) -> AgentState:
    """Decide which sub-agent or control node should act next.

    This node owns routing policy. The triage, action, HIL, dispatch, and comms
    nodes perform work, then always return here for the next decision.
    """
    proposed_action = state.get("proposed_action")

    if state.get("status") == "resolved":
        next_step: NextStep = "end" if state.get("comms_done") else "comms_agent"
    elif not state.get("report"):
        next_step = "triage_agent"
    elif not _should_act_from_triage(state):
        next_step = "comms_agent" if not state.get("comms_done") else "end"
    elif proposed_action is None:
        next_step = "action_agent"
    elif proposed_action == "no_action":
        next_step = "comms_agent" if not state.get("comms_done") else "end"
    elif state.get("awaiting_approval"):
        next_step = "pause_for_human"
    elif state.get("requires_hil") and not state.get("hil_approval_id"):
        next_step = "request_hil"
    elif not state.get("dispatch_status"):
        next_step = "dispatch"
    elif not state.get("comms_done"):
        next_step = "comms_agent"
    else:
        next_step = "end"

    log.info(
        "supervisor.route",
        investigation_id=state.get("investigation_id"),
        next_step=next_step,
    )
    return {**state, "next_step": next_step}


def route_from_supervisor(state: AgentState) -> NextStep:
    """Return the route selected by ``supervisor_node``."""
    return state.get("next_step", "end")


async def triage_agent_node(state: AgentState) -> AgentState:
    """Fetch drift report and run the triage sub-agent analysis."""
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


async def action_agent_node(state: AgentState) -> AgentState:
    """Propose replay_test, retrain, rollback, or no_action."""
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

    return {
        **state,
        "proposed_action": proposal.action,
        "action_rationale": proposal.rationale,
        "requires_hil": proposal.requires_hil,
        "awaiting_approval": False,
    }


async def request_hil_node(state: AgentState) -> AgentState:
    """Create the HIL approval request before the graph interrupt."""
    proposed_action = state.get("proposed_action")
    if proposed_action not in ("replay_test", "retrain", "rollback"):
        return {**state, "requires_hil": False, "awaiting_approval": False}

    hil_tool = TOOL_REGISTRY["request_hil_approval"]
    hil_result = await hil_tool.safe_run(
        {
            "investigation_id": state["investigation_id"],
            "action": proposed_action,
            "rationale": state.get("action_rationale")
            or state.get("triage_notes")
            or "Action requires approval.",
            "model_version": state["alert"].get("model_version", 1),
        }
    )
    if not hil_result.ok:
        return {
            **state,
            "summary_md": f"HIL approval request failed: {hil_result.error}",
            "status": "resolved",
        }

    return {
        **state,
        "requires_hil": True,
        "awaiting_approval": True,
        "hil_approval_id": hil_result.result.approval_id,  # type: ignore[union-attr]
        "status": "awaiting_approval",
    }


async def pause_for_human_node(state: AgentState) -> AgentState:
    """Interrupt/resume point that validates the recorded HIL approval."""
    approval_ok = await _approval_is_valid(state)
    if not approval_ok:
        return {
            **state,
            "awaiting_approval": False,
            "status": "resolved",
            "summary_md": "HIL approval is missing, rejected, or does not match the proposal.",
        }

    return {**state, "awaiting_approval": False, "status": "open"}


async def comms_agent_node(state: AgentState) -> AgentState:
    """Compose and persist the comms sub-agent dashboard summary."""

    if not state.get("report"):
        return {**state, "comms_done": True}

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
    return {**state, "summary_md": summary_md, "comms_done": True}


async def dispatch_node(state: AgentState) -> AgentState:
    """Validate approval/freshness, then enqueue the action to the arq worker."""
    proposed_action = state.get("proposed_action")
    if (
        state.get("awaiting_approval")
        or proposed_action not in ("replay_test", "retrain", "rollback")
    ):
        return state

    if state.get("requires_hil") and not await _approval_is_valid(state):
        return {
            **state,
            "dispatch_status": "blocked",
            "status": "resolved",
            "summary_md": "Dispatch blocked because HIL approval is invalid.",
        }

    stale_error = await _freshness_error(state)
    if stale_error:
        log.warning("dispatch.stale", error=stale_error)
        return {
            **state,
            "dispatch_status": "blocked",
            "status": "resolved",
            "summary_md": stale_error,
        }

    try:
        model_version = await _reconciled_model_version(state)
    except Exception as exc:
        log.warning("dispatch.reconcile_failed", error=str(exc))
        return {
            **state,
            "dispatch_status": "blocked",
            "status": "resolved",
            "summary_md": f"Dispatch blocked because model version reconciliation failed: {exc}",
        }

    dispatch_tool = TOOL_REGISTRY["dispatch_action"]
    result = await dispatch_tool.safe_run(
        {
            "investigation_id": state["investigation_id"],
            "action": proposed_action,
            "hil_approval_id": state.get("hil_approval_id") or "",
            "model_version": model_version,
        }
    )
    status = result.result.status if result.ok else "error"  # type: ignore[union-attr]
    return {**state, "dispatch_status": status, "status": "resolved"}


def _should_act_from_triage(state: AgentState) -> bool:
    severity = state["alert"].get("severity", "low")
    return severity in ("medium", "high")


async def _approval_is_valid(state: AgentState) -> bool:
    """Verify the approval row matches this investigation and proposal."""
    approval_id = state.get("hil_approval_id")
    proposed_action = state.get("proposed_action")
    if not approval_id or proposed_action not in ("replay_test", "retrain", "rollback"):
        return False

    from sqlalchemy import text

    from agent.deps.db import get_session

    async with get_session() as session:
        result = await session.execute(
            text(
                "SELECT status, decision, action, model_version "
                "FROM hil_approvals "
                "WHERE id = :id AND investigation_id = :investigation_id"
            ),
            {
                "id": approval_id,
                "investigation_id": state["investigation_id"],
            },
        )
        row = result.fetchone()

    if row is None:
        return False

    approval = row._mapping
    return (
        approval["status"] == "approved"
        and approval["decision"] == "approved"
        and approval["action"] == proposed_action
        and approval["model_version"] == state["alert"].get("model_version", 1)
    )


async def _freshness_error(state: AgentState) -> str | None:
    """Re-check staleness against the latest drift report before dispatch."""
    if not state.get("drift_report_id"):
        return None

    from agent.staleness import assert_not_stale

    tool = TOOL_REGISTRY["inspect_drift"]
    result = await tool.safe_run({"model_name": state["alert"].get("model_name", "")})
    if not result.ok:
        return f"Could not refresh drift report before dispatch: {result.error}"

    current_report: DriftReport = result.result.report  # type: ignore[union-attr]
    try:
        await assert_not_stale(state["drift_report_id"], current_report)
    except Exception as exc:
        return str(exc)
    return None


async def _reconciled_model_version(state: AgentState) -> int:
    """Return a model version that still resolves after checkpoint resume."""
    from agent.reconcile import reconcile_model_uri

    model_name = state["alert"].get("model_name", "")
    model_version = state["alert"].get("model_version", 1)
    if not model_name:
        return model_version
    return await reconcile_model_uri(model_name, model_version)


# Backward-compatible names for tests/imports that predate the explicit
# supervisor/sub-agent topology.
triage_node = triage_agent_node
action_node = action_agent_node
comms_node = comms_agent_node


def build_graph() -> Any:
    """Construct the LangGraph supervisor graph with Postgres checkpointer."""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    from core.settings import get_settings

    settings = get_settings()
    checkpointer = AsyncPostgresSaver.from_conn_string(settings.sync_database_url)
    checkpointer.setup()

    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("triage_agent", triage_agent_node)
    graph.add_node("action_agent", action_agent_node)
    graph.add_node("request_hil", request_hil_node)
    graph.add_node("pause_for_human", pause_for_human_node)  # interrupt point
    graph.add_node("dispatch", dispatch_node)
    graph.add_node("comms_agent", comms_agent_node)

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "triage_agent": "triage_agent",
            "action_agent": "action_agent",
            "request_hil": "request_hil",
            "pause_for_human": "pause_for_human",
            "dispatch": "dispatch",
            "comms_agent": "comms_agent",
            "end": END,
        },
    )
    graph.add_edge("triage_agent", "supervisor")
    graph.add_edge("action_agent", "supervisor")
    graph.add_edge("request_hil", "supervisor")
    graph.add_edge("pause_for_human", "supervisor")
    graph.add_edge("dispatch", "supervisor")
    graph.add_edge("comms_agent", "supervisor")

    return graph.compile(checkpointer=checkpointer, interrupt_before=["pause_for_human"])
