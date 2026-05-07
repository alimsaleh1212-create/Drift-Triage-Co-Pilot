"""Drift computation tests: known distributions produce known PSI / chi² values."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from drift.chi2 import chi2_result
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
    cur = pd.Series(rng.normal(0.4, 1, 500))
    psi = compute_psi(ref, cur)
    assert 0.05 <= psi <= 0.5  # broad range for moderate shift


def test_psi_raises_on_empty_series() -> None:
    with pytest.raises(ValueError):
        compute_psi(pd.Series(dtype=float), pd.Series([1.0, 2.0]))


def test_severity_max_aggregation() -> None:
    from drift.chi2 import Chi2Result
    from drift.output_drift import OutputDriftResult
    from drift.psi import PSIResult

    psi = [
        PSIResult(feature="a", psi=0.05, severity="low", reference_n=100, current_n=100)
    ]
    chi2 = [
        Chi2Result(
            feature="b",
            statistic=20.0,
            p_value=0.001,
            dof=5,
            severity="high",
            reference_n=100,
            current_n=100,
        )
    ]
    od = OutputDriftResult(
        psi=0.08,
        severity="low",
        reference_class_1_rate=0.1,
        current_class_1_rate=0.1,
        current_n=100,
    )
    assert aggregate_severity(psi, chi2, od) == "high"


def test_severity_all_low() -> None:
    from drift.chi2 import Chi2Result
    from drift.output_drift import OutputDriftResult
    from drift.psi import PSIResult

    psi = [
        PSIResult(feature="a", psi=0.02, severity="low", reference_n=100, current_n=100)
    ]
    chi2 = [
        Chi2Result(
            feature="b",
            statistic=2.0,
            p_value=0.3,
            dof=5,
            severity="low",
            reference_n=100,
            current_n=100,
        )
    ]
    od = OutputDriftResult(
        psi=0.01,
        severity="low",
        reference_class_1_rate=0.1,
        current_class_1_rate=0.1,
        current_n=100,
    )
    assert aggregate_severity(psi, chi2, od) == "low"


def test_chi2_result_accepts_alpha_parameter() -> None:
    ref_freqs = {"admin.": 0.25, "blue-collar": 0.22, "technician": 0.16}
    current = pd.Series(["admin."] * 50 + ["blue-collar"] * 40 + ["technician"] * 10)
    result = chi2_result("job", ref_freqs, current, alpha=0.05)
    assert result.feature == "job"
    assert result.current_n == 100


def test_chi2_result_high_severity_with_low_alpha() -> None:
    ref_freqs = {"a": 0.5, "b": 0.5}
    current = pd.Series(["a"] * 90 + ["b"] * 10)
    result = chi2_result("feat", ref_freqs, current, alpha=0.01)
    assert result.severity in ("low", "medium", "high")


@pytest.mark.asyncio
async def test_fetch_rolling_window_includes_prediction_labels() -> None:
    from service.routers.drift import _fetch_rolling_window

    result = MagicMock()
    result.fetchall.return_value = [
        SimpleNamespace(features={"age": 35, "job": "admin."}, label=1),
        SimpleNamespace(features='{"age": 41, "job": "technician"}', label=0),
    ]
    session = MagicMock()
    session.execute = AsyncMock(return_value=result)

    df = await _fetch_rolling_window(session, model_name="model", window_size=500)

    assert df[["age", "job", "label"]].to_dict(orient="records") == [
        {"age": 35, "job": "admin.", "label": 1},
        {"age": 41, "job": "technician", "label": 0},
    ]
