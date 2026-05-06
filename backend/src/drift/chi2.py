"""Chi-squared test for categorical feature drift detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy.stats import chi2_contingency  # type: ignore[import-untyped]

from core.settings import get_settings


class Chi2Result(BaseModel):
    """Chi-squared test result for one categorical feature."""

    feature: str
    statistic: float = Field(..., ge=0.0)
    p_value: float = Field(..., ge=0.0, le=1.0)
    dof: int
    severity: str  # "low" | "medium" | "high"
    reference_n: int
    current_n: int


def _chi2_severity(p_value: float, alpha: float) -> str:
    if p_value < 0.01:
        return "high"
    if p_value < alpha:
        return "medium"
    return "low"


def chi2_result(
    feature: str,
    reference_freqs: dict[str, float],
    current: pd.Series,
    alpha: float | None = None,
) -> Chi2Result:
    """Chi-squared test comparing current category counts to reference frequencies.

    Args:
        feature: Column name.
        reference_freqs: Relative frequency dict from ReferenceStats.categorical.
        current: Current rolling-window series of category values.
        alpha: Significance threshold (defaults to Settings.drift_chi2_alpha).

    Returns:
        Chi2Result with test statistic, p-value, and severity.
    """
    if alpha is None:
        alpha = get_settings().drift_chi2_alpha
    all_cats = sorted(reference_freqs.keys())

    ref_n = sum(reference_freqs.values())
    cur_counts = current.value_counts()

    # Build aligned count vectors; unseen categories get count 0
    cur_total = len(current)
    observed = np.array([cur_counts.get(cat, 0) for cat in all_cats], dtype=float)
    # Scale reference proportions to the current window size for chi2
    expected = np.array(
        [reference_freqs.get(cat, 1e-8) * cur_total for cat in all_cats],
        dtype=float,
    )

    # Avoid zero expected counts (chi2 undefined)
    expected = np.where(expected < 1e-8, 1e-8, expected)

    # Manually compute chi2 since we have pre-binned expected
    chi2_stat = float(np.sum((observed - expected) ** 2 / expected))
    dof = max(len(all_cats) - 1, 1)

    from scipy.stats import chi2 as chi2_dist

    p_value = float(1.0 - chi2_dist.cdf(chi2_stat, dof))

    return Chi2Result(
        feature=feature,
        statistic=chi2_stat,
        p_value=p_value,
        dof=dof,
        severity=_chi2_severity(p_value, alpha),
        reference_n=int(ref_n * cur_total),  # approximation for logging
        current_n=cur_total,
    )
