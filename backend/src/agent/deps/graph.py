"""Graph dependency: pull the compiled LangGraph from app.state."""

from __future__ import annotations

import os
from typing import Any

from fastapi import HTTPException, Request
from langgraph.graph.state import CompiledStateGraph

from core.logging import get_logger

log = get_logger(__name__)


def get_graph(request: Request) -> CompiledStateGraph:
    """Return the compiled LangGraph agent loaded during lifespan."""
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(
            status_code=503,
            detail="Agent graph not initialised. Wait for startup to complete.",
        )
    return graph


def _trace_tags(extra: list[str] | None = None) -> list[str]:
    """Build LangSmith trace tags from environment and caller-supplied extras."""
    tags = ["drift-triage", "langgraph", "supervisor"]
    environment = os.getenv("APP_ENV") or os.getenv("ENVIRONMENT")
    if environment:
        tags.append(environment)
    if extra:
        tags.extend(extra)
    return tags


def graph_run_config(
    *,
    investigation_id: str,
    payload: Any,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Build LangGraph RunnableConfig for a new investigation run."""
    from drift.severity import DriftWebhookPayload

    if isinstance(payload, DriftWebhookPayload):
        report_id = payload.report_id
        model_name = payload.model_name
        model_version = payload.model_version
    else:
        report_id = None
        model_name = None
        model_version = None

    metadata = {
        "investigation_id": investigation_id,
        "report_id": report_id,
        "drift_severity": getattr(payload, "severity", None),
        "model_name": model_name,
        "model_version": model_version,
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


def graph_resume_config(*, investigation_id: str) -> dict[str, Any]:
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


def request_id_from_request(request: Any) -> str | None:
    """Extract request/correlation ID from incoming request headers."""
    if request is None:
        return None
    return request.headers.get("x-request-id") or request.headers.get(
        "x-correlation-id"
    )
