"""POST /webhook/drift — receive drift severity change webhooks."""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from agent.deps.db import get_session
from agent.deps.graph import get_graph, graph_run_config, request_id_from_request
from core.logging import get_logger
from drift.severity import DriftWebhookPayload

router = APIRouter()
log = get_logger(__name__)


class WebhookResponse(BaseModel):
    investigation_id: str
    status: str


@router.post("/webhook/drift", response_model=WebhookResponse)
async def receive_drift_webhook(
    payload: DriftWebhookPayload,
    background_tasks: BackgroundTasks,
    request: Request,
    graph: CompiledStateGraph = Depends(get_graph),
) -> WebhookResponse:
    """Receive drift severity change webhook; open one investigation per event."""
    investigation_id = str(uuid4())

    investigation_id, created = await _create_investigation(investigation_id, payload)
    log.info(
        "webhook.received" if created else "webhook.duplicate",
        investigation_id=investigation_id,
        event_id=payload.event_id,
        severity=payload.severity,
        report_id=payload.report_id,
    )

    if not created:
        return WebhookResponse(investigation_id=investigation_id, status="open")

    async def _run_graph() -> None:
        initial_state = {
            "investigation_id": investigation_id,
            "alert": payload.model_dump(mode="json"),
            "report": None,
            "triage_notes": "",
            "proposed_action": None,
            "requires_hil": False,
            "hil_approval_id": None,
            "awaiting_approval": False,
            "summary_md": "",
            "dispatch_status": "",
            "status": "open",
            "drift_report_id": payload.report_id,
        }
        config = graph_run_config(
            investigation_id=investigation_id,
            payload=payload,
            request_id=request_id_from_request(request),
        )
        try:
            await graph.ainvoke(initial_state, config=config)
        except Exception as exc:
            log.exception(
                "graph.run_failed", investigation_id=investigation_id, error=str(exc)
            )

    background_tasks.add_task(_run_graph)
    return WebhookResponse(investigation_id=investigation_id, status="open")


def _drift_event_summary(payload: DriftWebhookPayload) -> str:
    """Build the initial dashboard summary for a newly opened investigation."""
    top_features = ", ".join(
        f"{item.feature}={item.value:g} ({item.severity})"
        for item in payload.top_features
    )
    if not top_features:
        top_features = "none"
    return (
        "Drift event received: "
        f"severity={payload.severity}, "
        f"previous_severity={payload.previous_severity or 'none'}, "
        f"model_name={payload.model_name}, "
        f"model_version={payload.model_version}, "
        f"report_id={payload.report_id}, "
        f"event_id={payload.event_id}, "
        f"top_features={top_features}. "
        f"{payload.drift_summary.text}"
    )


async def _create_investigation(
    investigation_id: str,
    payload: DriftWebhookPayload,
) -> tuple[str, bool]:
    """Persist the investigation row before graph/HIL work starts."""
    from sqlalchemy import text

    async with get_session() as session:
        existing = await _find_existing_investigation(session, payload)
        if existing:
            return existing, False

        await session.execute(
            text(
                "INSERT INTO investigations "
                "(id, drift_event_id, drift_report_id, status, summary_md,"
                " created_at, updated_at) "
                "VALUES (:id, :event_id, :report_id, :status, :summary_md,"
                " now(), now())"
            ),
            {
                "id": investigation_id,
                "event_id": payload.event_id,
                "report_id": payload.report_id,
                "status": "open",
                "summary_md": _drift_event_summary(payload),
            },
        )
        try:
            await session.commit()
        except IntegrityError:
            await session.rollback()
            existing = await _find_existing_investigation(session, payload)
            if existing:
                return existing, False
            raise
        return investigation_id, True


async def _find_existing_investigation(
    session: AsyncSession,
    payload: DriftWebhookPayload,
) -> str | None:
    """Return an existing investigation for this webhook event/report pair."""
    from sqlalchemy import text

    result = await session.execute(
        text(
            "SELECT id FROM investigations "
            "WHERE drift_event_id = :event_id OR drift_report_id = :report_id "
            "ORDER BY created_at ASC "
            "LIMIT 1"
        ),
        {"event_id": payload.event_id, "report_id": payload.report_id},
    )
    row = result.fetchone()
    return row.id if row else None
