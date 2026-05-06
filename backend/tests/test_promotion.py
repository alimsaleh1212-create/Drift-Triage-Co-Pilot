"""Tests for promotion gate: insufficient metrics → rejection, valid → promotion."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


def _make_model_version(stage: str, tags: dict | None = None):
    """Create a mock MlflowClient model version."""
    version = MagicMock()
    version.current_stage = stage
    version.version = "2"
    version.tags = tags or {}
    version.source = "models:/bank-marketing-classifier/2"
    version.run_id = "test_run_id"
    return version


@pytest.fixture()
def client():
    with patch("service.main.get_settings") as mock_settings, \
         patch("service.main.load_model") as mock_load, \
         patch("service.main.load_reference_stats"), \
         patch("service.main.create_async_engine") as mock_create_engine, \
         patch("service.main.async_sessionmaker") as mock_sessionmaker, \
         patch("service.routers.promotion.get_settings"):
        from core.settings import Settings

        settings = Settings(
            google_api_key="test-key",
            postgres_password="testpassword",
            promotion_api_key="correct_promo_key_16",
        )
        mock_settings.return_value = settings

        pipeline_mock = MagicMock()
        pipeline_mock.predict_proba = MagicMock(return_value=np.array([[0.25, 0.75]]))
        mock_load.return_value = (pipeline_mock, 0.5)
        engine = MagicMock()
        engine.dispose = AsyncMock()
        mock_create_engine.return_value = engine

        session = MagicMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        session.close = AsyncMock()
        mock_sessionmaker.return_value = MagicMock(return_value=session)

        with patch("service.routers.promotion.get_settings", return_value=settings):
            from service.main import app

            with TestClient(app) as c:
                yield c


def test_promotion_rejects_wrong_api_key(client):
    response = client.post(
        "/api/v1/promotion/promote",
        json={
            "model_name": "bank-marketing-classifier",
            "target_version": 2,
            "investigation_id": "inv-1",
            "hil_approval_id": "hil-1",
        },
        headers={"X-Promotion-Key": "wrong_key"},
    )
    assert response.status_code == 403


def test_promotion_rejects_missing_api_key(client):
    response = client.post(
        "/api/v1/promotion/promote",
        json={
            "model_name": "bank-marketing-classifier",
            "target_version": 2,
            "investigation_id": "inv-1",
            "hil_approval_id": "hil-1",
        },
    )
    assert response.status_code == 403


def test_promotion_rejects_low_recall(client):
    with patch("mlflow.MlflowClient") as mock_mlflow_client, \
         patch("mlflow.set_tracking_uri"):
        mock_client = MagicMock()
        target = _make_model_version("Staging", tags={"auc": "0.85", "recall": "0.60"})
        prod = _make_model_version("Production", tags={"auc": "0.83"})
        mock_client.get_model_version.return_value = target
        mock_client.get_latest_versions.return_value = [prod]
        mock_mlflow_client.return_value = mock_client

        response = client.post(
            "/api/v1/promotion/promote",
            json={
                "model_name": "bank-marketing-classifier",
                "target_version": 2,
                "investigation_id": "inv-1",
                "hil_approval_id": "hil-1",
            },
            headers={"X-Promotion-Key": "correct_promo_key_16"},
        )
        assert response.status_code == 400
        assert "recall" in response.json()["detail"].lower()


def test_promotion_rejects_lower_auc(client):
    with patch("mlflow.MlflowClient") as mock_mlflow_client, \
         patch("mlflow.set_tracking_uri"):
        mock_client = MagicMock()
        target = _make_model_version("Staging", tags={"auc": "0.80", "recall": "0.80"})
        prod = _make_model_version("Production", tags={"auc": "0.85"})
        mock_client.get_model_version.return_value = target
        mock_client.get_latest_versions.return_value = [prod]
        mock_mlflow_client.return_value = mock_client

        response = client.post(
            "/api/v1/promotion/promote",
            json={
                "model_name": "bank-marketing-classifier",
                "target_version": 2,
                "investigation_id": "inv-1",
                "hil_approval_id": "hil-1",
            },
            headers={"X-Promotion-Key": "correct_promo_key_16"},
        )
        assert response.status_code == 400
        assert "auc" in response.json()["detail"].lower()


def test_drift_webhook_payload_schema():
    from drift.severity import (
        DriftWebhookPayload,
        WebhookChi2Result,
        WebhookOutputDrift,
        WebhookPSIResult,
    )

    payload = DriftWebhookPayload(
        version="v1",
        report_id="rpt-123",
        model_name="bank-marketing-classifier",
        model_version=1,
        severity="high",
        psi_results=[WebhookPSIResult(feature="euribor3m", psi=0.35, severity="high")],
        chi2_results=[WebhookChi2Result(feature="job", statistic=25.0, p_value=0.003, severity="high")],
        output_drift=WebhookOutputDrift(psi=0.15, severity="medium"),
        timestamp="2025-01-01T00:00:00Z",
        window_size=500,
    )
    assert payload.version == "v1"
    assert payload.severity == "high"

    with pytest.raises(Exception):
        DriftWebhookPayload(
            version="v2",
            report_id="rpt-123",
            model_name="m",
            model_version=1,
            severity="critical",
            psi_results=[],
            chi2_results=[],
            output_drift=WebhookOutputDrift(psi=0.0, severity="low"),
            timestamp="2025-01-01T00:00:00Z",
            window_size=100,
        )
