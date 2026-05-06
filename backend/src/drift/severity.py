"""Aggregate drift severity across all feature tests and versioned webhook contract."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from drift.chi2 import Chi2Result
from drift.output_drift import OutputDriftResult
from drift.psi import PSIResult

Severity = Literal["low", "medium", "high"]

_SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2}


class WebhookPSIResult(BaseModel):
    """PSI result within a webhook payload (contract v1)."""

    feature: str
    psi: float = Field(..., ge=0.0)
    severity: Severity


class WebhookChi2Result(BaseModel):
    """Chi-squared result within a webhook payload (contract v1)."""

    feature: str
    statistic: float = Field(..., ge=0.0)
    p_value: float = Field(..., ge=0.0, le=1.0)
    severity: Severity


class WebhookOutputDrift(BaseModel):
    """Output drift within a webhook payload (contract v1)."""

    psi: float = Field(..., ge=0.0)
    severity: Severity


class DriftWebhookPayload(BaseModel):
    """Versioned contract: platform → agent on drift severity change.

    Per CLAUDE.md §23, schema changes are breaking — version the contract.
    The JSON Schema is mirrored in contracts/v1/drift_webhook.json.
    """

    version: Literal["v1"] = "v1"
    report_id: str = Field(..., min_length=1)
    model_name: str = Field(..., min_length=1)
    model_version: int = Field(..., ge=1)
    severity: Severity
    psi_results: list[WebhookPSIResult]
    chi2_results: list[WebhookChi2Result]
    output_drift: WebhookOutputDrift
    timestamp: datetime
    window_size: int = Field(..., ge=0)


class DriftReport(BaseModel):
    """Complete drift report for a model at a point in time."""

    report_id: str = Field(default_factory=lambda: str(uuid4()))
    model_name: str
    model_version: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    psi_results: list[PSIResult]
    chi2_results: list[Chi2Result]
    output_drift: OutputDriftResult
    severity: Severity
    window_size: int


def aggregate_severity(
    psi_results: list[PSIResult],
    chi2_results: list[Chi2Result],
    output_drift: OutputDriftResult,
) -> Severity:
    """Return the maximum severity across all feature tests.

    Args:
        psi_results: PSI results for numeric features.
        chi2_results: Chi-squared results for categorical features.
        output_drift: PSI on output class distribution.

    Returns:
        Overall severity: max across all individual feature severities.
    """
    all_severities = (
        [r.severity for r in psi_results]
        + [r.severity for r in chi2_results]
        + [output_drift.severity]
    )
    return max(all_severities, key=lambda s: _SEVERITY_ORDER[s])  # type: ignore[return-value]


def build_drift_report(
    model_name: str,
    model_version: int,
    psi_results: list[PSIResult],
    chi2_results: list[Chi2Result],
    output_drift: OutputDriftResult,
    window_size: int,
) -> DriftReport:
    """Assemble a complete DriftReport from individual feature test results.

    Args:
        model_name: Registered MLflow model name.
        model_version: Registered MLflow model version integer.
        psi_results: PSI per numeric feature.
        chi2_results: Chi-squared per categorical feature.
        output_drift: Output distribution drift result.
        window_size: Number of predictions in the rolling window.

    Returns:
        DriftReport with aggregate severity.
    """
    severity = aggregate_severity(psi_results, chi2_results, output_drift)
    return DriftReport(
        model_name=model_name,
        model_version=model_version,
        psi_results=psi_results,
        chi2_results=chi2_results,
        output_drift=output_drift,
        severity=severity,
        window_size=window_size,
    )


def report_to_webhook(report: DriftReport) -> DriftWebhookPayload:
    """Convert a DriftReport to the versioned webhook contract payload.

    Per CLAUDE.md §23, the contract between platform and agent must be
    versioned and explicit. This function handles the mapping from the
    internal DriftReport model to the contract DriftWebhookPayload.
    """
    return DriftWebhookPayload(
        version="v1",
        report_id=report.report_id,
        model_name=report.model_name,
        model_version=report.model_version,
        severity=report.severity,
        psi_results=[
            WebhookPSIResult(feature=r.feature, psi=r.psi, severity=r.severity)
            for r in report.psi_results
        ],
        chi2_results=[
            WebhookChi2Result(
                feature=r.feature,
                statistic=r.statistic,
                p_value=r.p_value,
                severity=r.severity,
            )
            for r in report.chi2_results
        ],
        output_drift=WebhookOutputDrift(
            psi=report.output_drift.psi,
            severity=report.output_drift.severity,
        ),
        timestamp=report.timestamp,
        window_size=report.window_size,
    )
