"""Focused tests for agent webhook persistence."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import BackgroundTasks

from drift.severity import DriftWebhookPayload, WebhookOutputDrift


@pytest.mark.asyncio
async def test_receive_drift_webhook_creates_investigation_row() -> None:
    """Webhook intake inserts investigation before graph work can request HIL."""
    from agent.main import receive_drift_webhook

    payload = DriftWebhookPayload(
        version="v1",
        report_id="report-123",
        model_name="drift-triage-classifier",
        model_version=7,
        severity="high",
        psi_results=[],
        chi2_results=[],
        output_drift=WebhookOutputDrift(psi=0.31, severity="high"),
        timestamp=datetime.now(timezone.utc),
        window_size=500,
    )

    session = MagicMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()

    @asynccontextmanager
    async def fake_get_session():
        yield session

    with (
        patch("agent.main.uuid4", return_value="investigation-123"),
        patch("agent.main.get_session", fake_get_session),
    ):
        response = await receive_drift_webhook(payload, BackgroundTasks())

    assert response.investigation_id == "investigation-123"
    assert response.status == "open"

    statement, params = session.execute.await_args.args
    assert "INSERT INTO investigations" in str(statement)
    assert "(id, status, summary_md, created_at, updated_at)" in str(statement)
    assert params["id"] == "investigation-123"
    assert params["status"] == "open"
    assert "severity=high" in params["summary_md"]
    assert "model_name=drift-triage-classifier" in params["summary_md"]
    assert "model_version=7" in params["summary_md"]
    assert "report_id=report-123" in params["summary_md"]
    session.commit.assert_awaited_once()
