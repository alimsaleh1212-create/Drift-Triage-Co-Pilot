"""LangSmith tracing config for LangGraph invocations."""

from __future__ import annotations

from datetime import datetime, timezone

from drift.severity import DriftWebhookPayload, WebhookDriftSummary, WebhookOutputDrift


def test_graph_run_config_adds_safe_trace_metadata() -> None:
    from agent.main import _graph_run_config

    payload = DriftWebhookPayload(
        schema_version="v1",
        event_id="event-123",
        report_id="report-123",
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
        top_features=[],
        psi_results=[],
        chi2_results=[],
        output_drift=WebhookOutputDrift(psi=0.31, severity="high"),
        window_size=500,
    )

    config = _graph_run_config(
        investigation_id="investigation-123",
        payload=payload,
        request_id="request-abc",
    )

    assert config["configurable"] == {"thread_id": "investigation-123"}
    assert config["run_name"] == "drift-triage-langgraph"
    assert config["tags"][:3] == ["drift-triage", "langgraph", "supervisor"]
    assert config["metadata"] == {
        "investigation_id": "investigation-123",
        "report_id": "report-123",
        "drift_severity": "high",
        "model_name": "drift-triage-classifier",
        "model_version": 7,
        "request_id": "request-abc",
    }


def test_langsmith_settings_are_optional() -> None:
    from core.settings import Settings

    settings = Settings(
        google_api_key="test-google-api-key-for-testing",
        postgres_password="testpassword",
        promotion_api_key="test_promotion_key_16ch",
    )

    assert settings.langsmith_tracing is False
    assert settings.langsmith_api_key is None
    assert settings.langsmith_project == "drift-triage-copilot"
