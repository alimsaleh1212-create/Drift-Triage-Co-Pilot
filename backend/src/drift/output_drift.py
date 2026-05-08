"""Output distribution drift: PSI on predicted class proportions."""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field

from drift.psi import Severity, _psi_severity, compute_psi


class OutputDriftResult(BaseModel):
    """PSI on predicted class distribution (output drift)."""

    psi: float = Field(..., ge=0.0)
    severity: Severity
    reference_class_1_rate: float
    current_class_1_rate: float
    current_n: int


def compute_output_drift(
    reference_proportions: dict[str, float],
    current_predictions: pd.Series,
) -> OutputDriftResult:
    """Compute PSI between reference and current predicted class distributions.

    Args:
        reference_proportions: {"0": float, "1": float} from ReferenceStats.
        current_predictions: Series of predicted labels (0 or 1) from rolling window.

    Returns:
        OutputDriftResult with psi value and severity.
    """
    if len(current_predictions) == 0:
        return OutputDriftResult(
            psi=0.0,
            severity="low",
            reference_class_1_rate=reference_proportions.get("1", 0.0),
            current_class_1_rate=0.0,
            current_n=0,
        )

    cur_rate = float(current_predictions.mean())
    ref_rate = reference_proportions.get("1", 0.0)

    # Build binary series to reuse compute_psi
    ref_series = pd.Series([0.0] * 100 + [1.0] * int(ref_rate * 100))
    cur_series = current_predictions.astype(float)

    psi = compute_psi(ref_series, cur_series, bins=2)

    return OutputDriftResult(
        psi=psi,
        severity=_psi_severity(psi),
        reference_class_1_rate=ref_rate,
        current_class_1_rate=cur_rate,
        current_n=len(current_predictions),
    )
