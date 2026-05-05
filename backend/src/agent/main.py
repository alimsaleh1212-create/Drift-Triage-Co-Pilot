"""FastAPI agent service: webhook intake, HIL approval, investigations API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from agent.graph import build_graph
from core.logging import configure_logging, get_logger
from drift.severity import DriftReport

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    app.state.graph = build_graph()
    log.info("agent.startup")
    yield
    log.info("agent.shutdown")


app = FastAPI(title="Drift Triage — Agent", version="0.1.0", lifespan=lifespan)
router = APIRouter()


class WebhookResponse(BaseModel):
    investigation_id: str
    status: str


class HILApprovalRequest(BaseModel):
    investigation_id: str
    hil_approval_id: str
    decision: str  # "approved" | "rejected"


class HILApprovalResponse(BaseModel):
    investigation_id: str
    decision: str
    status: str


@router.post("/webhook/drift", response_model=WebhookResponse)
async def receive_drift_webhook(
    payload: DriftReport,
    background_tasks: BackgroundTasks,
    request: Any = None,
) -> WebhookResponse:
    """Receive drift severity change webhook; open a new investigation."""
    investigation_id = str(uuid4())
    log.info(
        "webhook.received",
        investigation_id=investigation_id,
        severity=payload.severity,
        report_id=payload.report_id,
    )

    async def _run_graph() -> None:
        graph = app.state.graph
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
        config = {"configurable": {"thread_id": investigation_id}}
        try:
            await graph.ainvoke(initial_state, config=config)
        except Exception as exc:
            log.exception("graph.run_failed", investigation_id=investigation_id, error=str(exc))

    background_tasks.add_task(_run_graph)
    return WebhookResponse(investigation_id=investigation_id, status="open")


@router.post("/hil/approve", response_model=HILApprovalResponse)
async def approve_hil(payload: HILApprovalRequest) -> HILApprovalResponse:
    """Record HIL decision and resume the paused investigation graph."""
    if payload.decision not in ("approved", "rejected"):
        raise HTTPException(status_code=400, detail="decision must be 'approved' or 'rejected'")

    log.info(
        "hil.decision",
        investigation_id=payload.investigation_id,
        decision=payload.decision,
    )

    if payload.decision == "approved":
        graph = app.state.graph
        config = {"configurable": {"thread_id": payload.investigation_id}}
        # Resume from the pause_for_human interrupt
        await graph.ainvoke(None, config=config)

    return HILApprovalResponse(
        investigation_id=payload.investigation_id,
        decision=payload.decision,
        status="resumed" if payload.decision == "approved" else "rejected",
    )


@router.get("/investigations")
async def list_investigations() -> list[dict[str, Any]]:
    """List recent investigations from Postgres."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from core.settings import get_settings

    settings = get_settings()
    engine = create_async_engine(settings.async_database_url)
    async with engine.begin() as conn:
        result = await conn.execute(
            text("SELECT id, status, summary_md, updated_at FROM investigations ORDER BY updated_at DESC LIMIT 50")
        )
        rows = [dict(r._mapping) for r in result]
    await engine.dispose()
    return rows


@router.get("/investigations/{investigation_id}")
async def get_investigation(investigation_id: str) -> dict[str, Any]:
    """Return a single investigation by ID."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from core.settings import get_settings

    settings = get_settings()
    engine = create_async_engine(settings.async_database_url)
    async with engine.begin() as conn:
        result = await conn.execute(
            text("SELECT * FROM investigations WHERE id = :id"),
            {"id": investigation_id},
        )
        row = result.fetchone()
    await engine.dispose()
    if not row:
        raise HTTPException(status_code=404, detail="Investigation not found")
    return dict(row._mapping)


app.include_router(router, tags=["agent"])


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
