"""Tests for inference helpers and drift webhook payload conversion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from drift.severity import (
    DriftReport,
    DriftWebhookPayload,
    WebhookChi2Result,
    WebhookOutputDrift,
    WebhookPSIResult,
    report_to_webhook,
)
from drift.output_drift import OutputDriftResult
from drift.psi import PSIResult
from ml.inference import predict_batch


class TestPredictBatch:
    def test_predict_batch_returns_probabilities(self):
        X = pd.DataFrame({"feat": [1.0, 2.0, 3.0]})
        pipeline = Pipeline([("clf", DummyClassifier(strategy="most_frequent"))])
        pipeline.fit(X, [0, 0, 1])
        result = predict_batch(X, pipeline)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        assert all(0 <= p <= 1 for p in result)

    def test_predict_batch_positive_class(self):
        X = pd.DataFrame({"feat": [1.0, 2.0]})
        pipeline = Pipeline([("clf", DummyClassifier(strategy="most_frequent"))])
        pipeline.fit(X, [0, 1])
        result = predict_batch(X, pipeline)
        assert result.shape == (2,)


class TestWebhookConversion:
    def _make_report(self) -> DriftReport:
        from datetime import datetime, timezone

        return DriftReport(
            model_name="bank-marketing-classifier",
            model_version=1,
            psi_results=[
                PSIResult(feature="euribor3m", psi=0.35, severity="high", reference_n=1000, current_n=500)
            ],
            chi2_results=[],
            output_drift=OutputDriftResult(
                psi=0.15,
                severity="medium",
                reference_class_1_rate=0.11,
                current_class_1_rate=0.20,
                current_n=500,
            ),
            severity="high",
            window_size=500,
        )

    def test_report_to_webhook_preserves_fields(self):
        report = self._make_report()
        webhook = report_to_webhook(report)
        assert isinstance(webhook, DriftWebhookPayload)
        assert webhook.version == "v1"
        assert webhook.model_name == "bank-marketing-classifier"
        assert webhook.severity == "high"
        assert len(webhook.psi_results) == 1
        assert webhook.psi_results[0].feature == "euribor3m"
        assert webhook.psi_results[0].psi == 0.35

    def test_webhook_rejects_invalid_version(self):
        with pytest.raises(ValidationError):
            DriftWebhookPayload(
                version="v2",
                report_id="rpt-1",
                model_name="m",
                model_version=1,
                severity="low",
                psi_results=[],
                chi2_results=[],
                output_drift=WebhookOutputDrift(psi=0.0, severity="low"),
                timestamp="2025-01-01T00:00:00Z",
                window_size=100,
            )

    def test_webhook_rejects_invalid_severity(self):
        with pytest.raises(ValidationError):
            DriftWebhookPayload(
                version="v1",
                report_id="rpt-1",
                model_name="m",
                model_version=1,
                severity="critical",
                psi_results=[],
                chi2_results=[],
                output_drift=WebhookOutputDrift(psi=0.0, severity="low"),
                timestamp="2025-01-01T00:00:00Z",
                window_size=100,
            )
