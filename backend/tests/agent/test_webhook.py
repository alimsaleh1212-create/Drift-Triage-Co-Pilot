"""Focused tests for agent webhook persistence."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import BackgroundTasks
from httpx import ASGITransport, AsyncClient

from drift.severity import (
    DriftWebhookPayload,
    WebhookDriftSummary,
    WebhookOutputDrift,
    WebhookTopFeature,
)


def _payload(
    *,
    event_id: str = "event-123",
    report_id: str = "report-123",
) -> DriftWebhookPayload:
    return DriftWebhookPayload(
        schema_version="v1",
        event_id=event_id,
        report_id=report_id,
        previous_severity="medium",
        model_name="drift-triage-classifier",
        model_version=7,
        severity="high",
        created_at=datetime.now(timezone.utc),
        drift_summary=WebhookDriftSummary(
            text="Severity changed to high.",
            window_size=500,
            output_drift_severity="high",
        ),
        top_features=[
            WebhookTopFeature(
                feature="euribor3m",
                metric="psi",
                value=0.31,
                severity="high",
            )
        ],
        psi_results=[],
        chi2_results=[],
        output_drift=WebhookOutputDrift(psi=0.31, severity="high"),
        window_size=500,
    )


class _Result:
    def __init__(self, row=None) -> None:
        self._row = row

    def fetchone(self):
        return self._row


class _Rows:
    def __init__(self, rows) -> None:
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


@pytest.mark.asyncio
async def test_receive_drift_webhook_creates_investigation_row() -> None:
    """Webhook intake inserts investigation before graph work can request HIL."""
    from agent.routers.webhook import receive_drift_webhook

    payload = _payload()

    session = MagicMock()
    session.execute = AsyncMock(side_effect=[_Result(), _Result()])
    session.commit = AsyncMock()

    @asynccontextmanager
    async def fake_get_session():
        yield session

    mock_graph = SimpleNamespace(ainvoke=AsyncMock())

    with (
        patch("agent.routers.webhook.uuid4", return_value="investigation-123"),
        patch("agent.routers.webhook.get_session", fake_get_session),
        patch("agent.routers.webhook.get_graph", return_value=mock_graph),
    ):
        response = await receive_drift_webhook(
            payload, BackgroundTasks(), request=None, graph=mock_graph
        )

    assert response.investigation_id == "investigation-123"
    assert response.status == "open"

    statement, params = session.execute.await_args_list[1].args
    assert "INSERT INTO investigations" in str(statement)
    assert "drift_event_id" in str(statement)
    assert "drift_report_id" in str(statement)
    assert params["id"] == "investigation-123"
    assert params["event_id"] == "event-123"
    assert params["report_id"] == "report-123"
    assert params["status"] == "open"
    assert "severity=high" in params["summary_md"]
    assert "previous_severity=medium" in params["summary_md"]
    assert "model_name=drift-triage-classifier" in params["summary_md"]
    assert "model_version=7" in params["summary_md"]
    assert "report_id=report-123" in params["summary_md"]
    assert "event_id=event-123" in params["summary_md"]
    assert "euribor3m" in params["summary_md"]
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_receive_drift_webhook_is_idempotent_for_duplicate_event() -> None:
    """Repeated delivery returns the original investigation without inserting."""
    from agent.routers.webhook import receive_drift_webhook

    payload = _payload(event_id="event-dup", report_id="report-dup")
    session = MagicMock()
    session.execute = AsyncMock(
        return_value=_Result(SimpleNamespace(id="investigation-existing"))
    )
    session.commit = AsyncMock()

    @asynccontextmanager
    async def fake_get_session():
        yield session

    mock_graph = SimpleNamespace(ainvoke=AsyncMock())

    with (
        patch("agent.routers.webhook.uuid4", return_value="investigation-new"),
        patch("agent.routers.webhook.get_session", fake_get_session),
        patch("agent.routers.webhook.get_graph", return_value=mock_graph),
    ):
        response = await receive_drift_webhook(
            payload, BackgroundTasks(), request=None, graph=mock_graph
        )

    assert response.investigation_id == "investigation-existing"
    assert response.status == "open"
    assert session.execute.await_count == 1
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_webhook_endpoint_accepts_valid_payload_and_runs_graph() -> None:
    """HTTP intake validates the payload, persists once, and schedules graph work."""
    from agent.main import app

    payload = _payload(event_id="event-http", report_id="report-http")
    session = MagicMock()
    session.execute = AsyncMock(side_effect=[_Result(), _Result()])
    session.commit = AsyncMock()

    @asynccontextmanager
    async def fake_get_session():
        yield session

    graph = SimpleNamespace(ainvoke=AsyncMock())
    app.state.graph = graph
    transport = ASGITransport(app=app)

    with (
        patch("agent.routers.webhook.uuid4", return_value="investigation-http"),
        patch("agent.routers.webhook.get_session", fake_get_session),
    ):
        async with AsyncClient(transport=transport, base_url="http://agent") as client:
            response = await client.post(
                "/webhook/drift",
                json=payload.model_dump(mode="json"),
            )

    assert response.status_code == 200
    assert response.json() == {
        "investigation_id": "investigation-http",
        "status": "open",
    }
    graph.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_list_investigations_returns_json_with_nullable_drift_ids() -> None:
    """Investigation listing enters the DB context and serializes legacy rows."""
    from agent.main import app

    session = MagicMock()
    session.execute = AsyncMock(
        return_value=_Rows(
            [
                SimpleNamespace(
                    _mapping={
                        "id": "investigation-legacy",
                        "drift_event_id": None,
                        "drift_report_id": None,
                        "status": "open",
                        "summary_md": None,
                        "updated_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
                    }
                )
            ]
        )
    )

    @asynccontextmanager
    async def fake_get_session():
        yield session

    transport = ASGITransport(app=app)

    with patch("agent.routers.investigations.get_session", fake_get_session):
        async with AsyncClient(transport=transport, base_url="http://agent") as client:
            health = await client.get("/health")
            response = await client.get("/investigations")

    assert health.status_code == 200
    assert response.status_code == 200
    assert response.json() == [
        {
            "id": "investigation-legacy",
            "drift_event_id": None,
            "drift_report_id": None,
            "status": "open",
            "summary_md": None,
            "updated_at": "2026-01-01T00:00:00Z",
        }
    ]
    session.execute.assert_awaited_once()
