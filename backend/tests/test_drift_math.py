"""Drift computation tests: known distributions produce known PSI / chi² values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drift.psi import compute_psi, psi_result
from drift.severity import aggregate_severity


def test_psi_identical_distributions_near_zero() -> None:
    rng = np.random.default_rng(42)
    data = pd.Series(rng.normal(0, 1, 1000))
    psi = compute_psi(data, data.copy())
    assert psi < 0.01


def test_psi_large_shift_is_high() -> None:
    rng = np.random.default_rng(42)
    ref = pd.Series(rng.normal(0, 1, 1000))
    cur = pd.Series(rng.normal(5, 1, 500))  # +5σ shift
    psi = compute_psi(ref, cur)
    assert psi >= 0.25


def test_psi_moderate_shift_is_medium() -> None:
    rng = np.random.default_rng(42)
    ref = pd.Series(rng.normal(0, 1, 1000))
    cur = pd.Series(rng.normal(0.8, 1, 500))
    psi = compute_psi(ref, cur)
    assert 0.05 <= psi <= 0.5  # broad range for moderate shift


def test_psi_raises_on_empty_series() -> None:
    with pytest.raises(ValueError):
        compute_psi(pd.Series(dtype=float), pd.Series([1.0, 2.0]))


def test_severity_max_aggregation() -> None:
    from drift.chi2 import Chi2Result
    from drift.output_drift import OutputDriftResult
    from drift.psi import PSIResult

    psi = [PSIResult(feature="a", psi=0.05, severity="low", reference_n=100, current_n=100)]
    chi2 = [Chi2Result(feature="b", statistic=20.0, p_value=0.001, dof=5, severity="high", reference_n=100, current_n=100)]
    od = OutputDriftResult(psi=0.08, severity="low", reference_class_1_rate=0.1, current_class_1_rate=0.1, current_n=100)
    assert aggregate_severity(psi, chi2, od) == "high"


def test_severity_all_low() -> None:
    from drift.chi2 import Chi2Result
    from drift.output_drift import OutputDriftResult
    from drift.psi import PSIResult

    psi = [PSIResult(feature="a", psi=0.02, severity="low", reference_n=100, current_n=100)]
    chi2 = [Chi2Result(feature="b", statistic=2.0, p_value=0.3, dof=5, severity="low", reference_n=100, current_n=100)]
    od = OutputDriftResult(psi=0.01, severity="low", reference_class_1_rate=0.1, current_class_1_rate=0.1, current_n=100)
    assert aggregate_severity(psi, chi2, od) == "low"
