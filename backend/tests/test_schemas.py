"""Schema tests: valid and invalid inputs for key Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from drift.chi2 import Chi2Result
from drift.output_drift import OutputDriftResult
from drift.psi import PSIResult
from drift.severity import (
    DriftReport,
    DriftWebhookPayload,
    WebhookChi2Result,
    WebhookOutputDrift,
    WebhookPSIResult,
    build_drift_report,
)
from service.routers.prediction import PredictRequest, PredictResponse
from service.routers.promotion import PromotionRequest


# ── PSI ─────────────────────────────────────────────────────────────────────

def test_psi_result_valid() -> None:
    r = PSIResult(feature="euribor3m", psi=0.05, severity="low", reference_n=1000, current_n=500)
    assert r.psi == 0.05


def test_psi_result_rejects_negative_psi() -> None:
    with pytest.raises(ValidationError):
        PSIResult(feature="x", psi=-0.1, severity="low", reference_n=100, current_n=100)


def test_psi_result_rejects_invalid_severity() -> None:
    with pytest.raises(ValidationError):
        PSIResult(feature="x", psi=0.1, severity="critical", reference_n=100, current_n=100)


# ── Chi2 ────────────────────────────────────────────────────────────────────

def test_chi2_result_valid() -> None:
    r = Chi2Result(feature="job", statistic=12.3, p_value=0.01, dof=9, severity="high", reference_n=1000, current_n=500)
    assert r.severity == "high"


def test_chi2_result_rejects_p_value_out_of_range() -> None:
    with pytest.raises(ValidationError):
        Chi2Result(feature="job", statistic=1.0, p_value=1.5, dof=5, severity="low", reference_n=100, current_n=100)


# ── DriftReport ─────────────────────────────────────────────────────────────

def test_drift_report_severity_aggregation() -> None:
    psi = [PSIResult(feature="f", psi=0.3, severity="high", reference_n=100, current_n=100)]
    chi2: list[Chi2Result] = []
    od = OutputDriftResult(psi=0.05, severity="low", reference_class_1_rate=0.1, current_class_1_rate=0.1, current_n=100)
    report = build_drift_report("m", 1, psi, chi2, od, 100)
    assert report.severity == "high"


# ── Prediction ───────────────────────────────────────────────────────────────

def test_predict_request_valid() -> None:
    req = PredictRequest(
        age=35, job="admin.", marital="married", education="university.degree",
        default="no", housing="yes", loan="no", contact="cellular",
        month="may", day_of_week="mon", campaign=1, pdays=999, previous=0,
        poutcome="nonexistent",
        **{"emp.var.rate": -1.8, "cons.price.idx": 92.9, "cons.conf.idx": -46.2,
           "euribor3m": 1.3, "nr.employed": 5099.1},
    )
    assert req.age == 35


def test_predict_request_rejects_invalid_age() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(
            age=15, job="admin.", marital="married", education="university.degree",
            default="no", housing="yes", loan="no", contact="cellular",
            month="may", day_of_week="mon", campaign=1, pdays=999, previous=0,
            poutcome="nonexistent",
            **{"emp.var.rate": -1.8, "cons.price.idx": 92.9, "cons.conf.idx": -46.2,
               "euribor3m": 1.3, "nr.employed": 5099.1},
        )


# ── Promotion ────────────────────────────────────────────────────────────────

def test_promotion_request_valid() -> None:
    r = PromotionRequest(
        model_name="drift-triage-classifier",
        target_version=2,
        investigation_id="inv-abc",
        hil_approval_id="hil-xyz",
    )
    assert r.target_version == 2


def test_promotion_request_rejects_zero_version() -> None:
    with pytest.raises(ValidationError):
        PromotionRequest(
            model_name="m", target_version=0, investigation_id="i", hil_approval_id="h"
        )


# ── DriftWebhookPayload ──────────────────────────────────────────────────────


def test_webhook_payload_valid() -> None:
    payload = DriftWebhookPayload(
        version="v1",
        report_id="rpt-abc",
        model_name="drift-triage-classifier",
        model_version=1,
        severity="high",
        psi_results=[WebhookPSIResult(feature="euribor3m", psi=0.35, severity="high")],
        chi2_results=[
            WebhookChi2Result(feature="job", statistic=25.0, p_value=0.003, severity="high")
        ],
        output_drift=WebhookOutputDrift(psi=0.15, severity="medium"),
        timestamp="2025-01-01T00:00:00Z",
        window_size=500,
    )
    assert payload.version == "v1"
    assert payload.severity == "high"


def test_webhook_payload_rejects_invalid_version() -> None:
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


def test_webhook_payload_rejects_invalid_severity() -> None:
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
