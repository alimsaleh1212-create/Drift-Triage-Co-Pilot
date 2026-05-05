"""Aggregate drift severity across all feature tests."""

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
