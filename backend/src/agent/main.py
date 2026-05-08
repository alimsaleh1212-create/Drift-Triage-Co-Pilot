"""FastAPI agent service: webhook intake, HIL approval, investigations API."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from agent.deps.db import close_db, get_session, init_db
from agent.graph import build_graph
from core.logging import configure_logging, get_logger
from core.settings import Settings
from drift.severity import DriftWebhookPayload

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    init_db()
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    from core.settings import get_settings

    settings = get_settings()
    _configure_langsmith(settings)
    async with AsyncPostgresSaver.from_conn_string(
        settings.checkpoint_database_url
    ) as checkpointer:
        await checkpointer.setup()
        app.state.graph = build_graph(checkpointer=checkpointer)
        log.info("agent.startup")
        yield
    await close_db()
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
    payload: DriftWebhookPayload,
    background_tasks: BackgroundTasks,
    request: Any = None,
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
        config = _graph_run_config(
            investigation_id=investigation_id,
            payload=payload,
            request_id=_request_id_from_request(request),
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


@router.post("/hil/approve", response_model=HILApprovalResponse)
async def approve_hil(payload: HILApprovalRequest) -> HILApprovalResponse:
    """Record HIL decision and resume the paused investigation graph."""
    if payload.decision not in ("approved", "rejected"):
        raise HTTPException(
            status_code=400, detail="decision must be 'approved' or 'rejected'"
        )

    log.info(
        "hil.decision",
        investigation_id=payload.investigation_id,
        decision=payload.decision,
    )

    async with get_session() as session:
        from sqlalchemy import text

        await session.execute(
            text(
                "UPDATE hil_approvals "
                "SET status = :status, decision = :decision, decided_at = now() "
                "WHERE id = :approval_id AND investigation_id = :investigation_id"
            ),
            {
                "status": payload.decision,
                "decision": payload.decision,
                "approval_id": payload.hil_approval_id,
                "investigation_id": payload.investigation_id,
            },
        )
        if payload.decision == "rejected":
            await session.execute(
                text(
                    "UPDATE investigations "
                    "SET status = 'resolved', updated_at = now() "
                    "WHERE id = :investigation_id"
                ),
                {"investigation_id": payload.investigation_id},
            )
        await session.commit()

    if payload.decision == "approved":
        graph = app.state.graph
        config = _graph_resume_config(investigation_id=payload.investigation_id)
        # Resume from the pause_for_human interrupt
        await graph.ainvoke(None, config=config)

    return HILApprovalResponse(
        investigation_id=payload.investigation_id,
        decision=payload.decision,
        status="resumed" if payload.decision == "approved" else "rejected",
    )


@router.get("/hil/approvals")
async def list_hil_approvals() -> list[dict[str, Any]]:
    """List pending HIL approvals with their real approval IDs."""
    from sqlalchemy import text

    async with get_session() as session:
        result = await session.execute(
            text(
                "SELECT "
                "h.id, h.investigation_id, h.action, h.rationale, h.model_version, "
                "h.status, h.created_at, i.summary_md, i.updated_at "
                "FROM hil_approvals h "
                "JOIN investigations i ON i.id = h.investigation_id "
                "WHERE h.status = 'pending' "
                "ORDER BY h.created_at DESC "
                "LIMIT 50"
            )
        )
        return [dict(r._mapping) for r in result]


@router.get("/queue/metrics")
async def queue_metrics() -> dict[str, Any]:
    """Return arq queue depth, DLQ size, dedup locks, and recent worker jobs."""
    import json

    import redis.asyncio as aioredis

    from core.settings import get_settings

    settings = get_settings()
    # decode_responses=True for string keys; arq result values are msgpack (binary)
    # so we use a separate bytes client to avoid UnicodeDecodeError on result scan.
    redis_str = aioredis.from_url(settings.redis_url, decode_responses=True)
    redis_bytes = aioredis.from_url(settings.redis_url, decode_responses=False)
    try:
        queue_key = f"arq:queue:{settings.redis_queue_name}"
        dlq_key = f"{settings.redis_queue_name}:dlq"
        # arq queue is a sorted set — never call llen on it (WRONGTYPE)
        queue_depth = await redis_str.zcard(queue_key)
        dlq_count = await redis_str.llen(dlq_key)
        active_dispatches = 0
        async for _ in redis_str.scan_iter(match="dispatch:*", count=100):
            active_dispatches += 1
        result_keys = []
        async for key in redis_bytes.scan_iter(match="arq:result:*", count=200):
            result_keys.append(key)
        recent_jobs: list[dict[str, Any]] = []
        for key in result_keys[-20:]:
            raw = await redis_bytes.get(key)
            if not raw:
                continue
            key_str = key.decode("utf-8") if isinstance(key, bytes) else key
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = {"raw": f"binary ({len(raw)} bytes)"}
            recent_jobs.append({"key": key_str, **parsed})
        dlq_items: list[dict[str, Any]] = []
        for raw in await redis_str.lrange(dlq_key, 0, 9):
            try:
                dlq_items.append(json.loads(raw))
            except Exception:
                dlq_items.append({"raw": raw})
        # Detect in-progress arq jobs: worker publishes heartbeat at arq:health-check
        # and marks running jobs at arq:{queue}:running (varies by version).
        # We approximate via "dispatch lock exists but queue is empty" = worker running.
        worker_running = bool(active_dispatches > 0 and queue_depth == 0)
        return {
            "queue_depth": int(queue_depth or 0),
            "dlq_count": int(dlq_count or 0),
            "active_dispatches": active_dispatches,
            "worker_running": worker_running,
            "recent_jobs_count": len(recent_jobs),
            "recent_jobs": recent_jobs,
            "dlq_items": dlq_items,
        }
    finally:
        await redis_str.aclose()
        await redis_bytes.aclose()


@router.post("/admin/reset")
async def admin_reset() -> dict[str, str]:
    """Mark all open investigations resolved and clear pending HIL approvals."""
    from sqlalchemy import text

    async with get_session() as session:
        await session.execute(
            text(
                "UPDATE investigations SET status = 'resolved',"
                " updated_at = now() WHERE status != 'resolved'"
            )
        )
        await session.execute(
            text(
                "UPDATE hil_approvals SET status = 'rejected',"
                " decision = 'rejected' WHERE status = 'pending'"
            )
        )
        await session.commit()
    log.info("admin.reset")
    return {"status": "reset"}


@router.get("/investigations")
async def list_investigations() -> list[dict[str, Any]]:
    """List recent investigations from Postgres."""
    from sqlalchemy import text

    async with get_session() as session:
        result = await session.execute(
            text(
                "SELECT "
                "id, drift_event_id, drift_report_id, status, summary_md, updated_at "
                "FROM investigations "
                "ORDER BY updated_at DESC "
                "LIMIT 50"
            )
        )
        return [dict(r._mapping) for r in result]


@router.get("/investigations/{investigation_id}")
async def get_investigation(
    investigation_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Return a single investigation by ID."""
    from sqlalchemy import text

    result = await session.execute(
        text("SELECT * FROM investigations WHERE id = :id"),
        {"id": investigation_id},
    )
    row = result.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Investigation not found")
    return dict(row._mapping)


app.include_router(router, tags=["agent"])


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def _configure_langsmith(settings: Settings) -> None:
    """Expose Settings-backed LangSmith values to LangGraph's env-based tracer."""
    if not settings.langsmith_tracing:
        return

    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
    if settings.langsmith_api_key:
        os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)

    log.info(
        "langsmith.tracing_configured",
        project=settings.langsmith_project,
    )


def _graph_run_config(
    *,
    investigation_id: str,
    payload: DriftWebhookPayload,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Build LangGraph RunnableConfig with safe LangSmith metadata."""
    metadata = {
        "investigation_id": investigation_id,
        "report_id": payload.report_id,
        "drift_severity": payload.severity,
        "model_name": payload.model_name,
        "model_version": payload.model_version,
        "request_id": request_id,
    }
    return {
        "configurable": {"thread_id": investigation_id},
        "run_name": "drift-triage-langgraph",
        "tags": _trace_tags(),
        "metadata": {
            key: value for key, value in metadata.items() if value is not None
        },
    }


def _graph_resume_config(*, investigation_id: str) -> dict[str, Any]:
    """Build trace config for graph resume after HIL approval."""
    return {
        "configurable": {"thread_id": investigation_id},
        "run_name": "drift-triage-langgraph-resume",
        "tags": _trace_tags(extra=["hil-resume"]),
        "metadata": {
            "investigation_id": investigation_id,
            "event": "hil_approval_resume",
        },
    }


def _trace_tags(extra: list[str] | None = None) -> list[str]:
    tags = ["drift-triage", "langgraph", "supervisor"]
    environment = os.getenv("APP_ENV") or os.getenv("ENVIRONMENT")
    if environment:
        tags.append(environment)
    if extra:
        tags.extend(extra)
    return tags


def _request_id_from_request(request: Any) -> str | None:
    if request is None:
        return None
    return request.headers.get("x-request-id") or request.headers.get(
        "x-correlation-id"
    )
