"""Focused tests for durable drift webhook severity-change delivery."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from drift.output_drift import OutputDriftResult
from drift.severity import DriftReport


class _Result:
    def __init__(self, row=None) -> None:
        self._row = row

    def fetchone(self):
        return self._row


class _Session:
    def __init__(self, persisted_severity: str | None) -> None:
        self.persisted_severity = persisted_severity
        self.execute = AsyncMock(side_effect=self._execute)
        self.commit = AsyncMock()
        self.rollback = AsyncMock()
        self.upsert_params: dict | None = None

    async def _execute(self, statement, params=None):
        sql = str(statement)
        if "SELECT last_severity" in sql:
            row = (
                SimpleNamespace(
                    last_severity=self.persisted_severity,
                    last_report_id="previous-report",
                )
                if self.persisted_severity is not None
                else None
            )
            return _Result(row)
        if "INSERT INTO drift_alert_state" in sql:
            self.upsert_params = params
            return _Result()
        return _Result()


def _request(session: _Session) -> SimpleNamespace:
    class SessionLocal:
        async def __aenter__(self):
            return session

        async def __aexit__(self, exc_type, exc, tb):
            return None

    state = SimpleNamespace(SessionLocal=SessionLocal, http_client=object())
    return SimpleNamespace(app=SimpleNamespace(state=state))


def _report(*, report_id: str, severity: str) -> DriftReport:
    return DriftReport(
        report_id=report_id,
        model_name="drift-triage-classifier",
        model_version=1,
        timestamp=datetime.now(timezone.utc),
        psi_results=[],
        chi2_results=[],
        output_drift=OutputDriftResult(
            psi=0.2,
            severity=severity,
            reference_class_1_rate=0.1,
            current_class_1_rate=0.2,
            current_n=500,
        ),
        severity=severity,
        window_size=500,
    )


@pytest.mark.asyncio
async def test_first_drift_report_emits_webhook_and_stores_last_severity() -> None:
    from service.routers.drift import _maybe_emit_severity_webhook

    session = _Session(persisted_severity=None)
    emit = AsyncMock()

    with patch("service.routers.drift._emit_webhook", emit):
        await _maybe_emit_severity_webhook(
            _request(session),
            _report(report_id="report-1", severity="medium"),
        )

    emit.assert_awaited_once()
    payload = emit.await_args.args[1]
    assert payload["schema_version"] == "v1"
    assert payload["severity"] == "medium"
    assert payload["previous_severity"] is None
    assert payload["report_id"] == "report-1"
    assert session.upsert_params == {
        "key": "drift-triage-classifier:1",
        "severity": "medium",
        "report_id": "report-1",
    }
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_same_severity_again_does_not_emit_webhook() -> None:
    from service.routers.drift import _maybe_emit_severity_webhook

    session = _Session(persisted_severity="medium")
    emit = AsyncMock()

    with patch("service.routers.drift._emit_webhook", emit):
        await _maybe_emit_severity_webhook(
            _request(session),
            _report(report_id="report-2", severity="medium"),
        )

    emit.assert_not_awaited()
    assert session.upsert_params is None
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_changed_severity_emits_new_webhook() -> None:
    from service.routers.drift import _maybe_emit_severity_webhook

    session = _Session(persisted_severity="medium")
    emit = AsyncMock()

    with patch("service.routers.drift._emit_webhook", emit):
        await _maybe_emit_severity_webhook(
            _request(session),
            _report(report_id="report-3", severity="high"),
        )

    emit.assert_awaited_once()
    payload = emit.await_args.args[1]
    assert payload["previous_severity"] == "medium"
    assert payload["severity"] == "high"
    assert session.upsert_params["severity"] == "high"


@pytest.mark.asyncio
async def test_restart_safe_behavior_uses_persisted_severity_not_app_state() -> None:
    from service.routers.drift import _maybe_emit_severity_webhook

    session = _Session(persisted_severity="high")
    request = _request(session)
    assert not hasattr(request.app.state, "last_severity")
    emit = AsyncMock()

    with patch("service.routers.drift._emit_webhook", emit):
        await _maybe_emit_severity_webhook(
            request,
            _report(report_id="report-4", severity="high"),
        )

    emit.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_webhook_does_not_store_last_severity() -> None:
    from service.routers.drift import _maybe_emit_severity_webhook

    session = _Session(persisted_severity="low")
    emit = AsyncMock(side_effect=RuntimeError("agent unavailable"))

    with patch("service.routers.drift._emit_webhook", emit):
        await _maybe_emit_severity_webhook(
            _request(session),
            _report(report_id="report-5", severity="high"),
        )

    emit.assert_awaited_once()
    assert session.upsert_params is None
    session.commit.assert_not_awaited()
    session.rollback.assert_awaited_once()
