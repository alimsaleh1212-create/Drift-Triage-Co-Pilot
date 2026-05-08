"""Population Stability Index (PSI) for numeric feature drift detection."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

Severity = Literal["low", "medium", "high"]


class PSIResult(BaseModel):
    """PSI computation result for one numeric feature."""

    feature: str
    psi: float = Field(..., ge=0.0)
    severity: Severity
    reference_n: int
    current_n: int


_PSI_WARN = 0.1
_PSI_HIGH = 0.25


def _psi_severity(psi: float) -> Severity:
    if psi >= _PSI_HIGH:
        return "high"
    if psi >= _PSI_WARN:
        return "medium"
    return "low"


def compute_psi(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
    epsilon: float = 1e-8,
) -> float:
    """Compute PSI between reference and current distributions.

    Uses quantile bins from the reference distribution to avoid empty bins.

    Args:
        reference: Baseline numeric series from training data.
        current: Rolling-window numeric series from recent predictions.
        bins: Number of quantile bins.
        epsilon: Small constant to avoid log(0).

    Returns:
        PSI value. < 0.1 = stable, 0.1–0.25 = moderate, >= 0.25 = significant.

    Raises:
        ValueError: If either series is empty.
    """
    if len(reference) == 0 or len(current) == 0:
        raise ValueError("reference and current must be non-empty")

    breakpoints = np.nanquantile(reference, np.linspace(0, 1, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts, _ = np.histogram(reference.dropna(), bins=breakpoints)
    cur_counts, _ = np.histogram(current.dropna(), bins=breakpoints)

    ref_pct = ref_counts / (ref_counts.sum() + epsilon) + epsilon
    cur_pct = cur_counts / (cur_counts.sum() + epsilon) + epsilon

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_result(
    feature: str,
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
) -> PSIResult:
    """Compute PSI and wrap into a typed result for a single feature.

    Args:
        feature: Column name (for labelling the result).
        reference: Baseline series.
        current: Current rolling-window series.
        bins: Number of quantile bins.

    Returns:
        PSIResult with psi value and severity label.
    """
    psi = compute_psi(reference, current, bins=bins)
    return PSIResult(
        feature=feature,
        psi=psi,
        severity=_psi_severity(psi),
        reference_n=len(reference),
        current_n=len(current),
    )
