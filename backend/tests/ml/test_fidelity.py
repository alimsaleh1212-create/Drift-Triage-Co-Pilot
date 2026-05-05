"""Fidelity replay test: loaded model produces identical output to 1e-12 precision.

Runs without a running MLflow server — loads model directly from the joblib
artifact path written by register.py.

Gate: `uv run pytest tests/ml/test_fidelity.py` must pass after `make train`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIDELITY_FIXTURE = Path(__file__).parent / "fixtures" / "fidelity_input.json"
FIDELITY_EXPECTED = Path(__file__).parent / "fixtures" / "fidelity_expected.json"
MODEL_PATH = Path("artifacts/models/latest.joblib")


@pytest.mark.skipif(
    not MODEL_PATH.exists() or not FIDELITY_FIXTURE.exists(),
    reason="run `make train` and record fixtures first",
)
def test_fidelity_replay() -> None:
    """Predict on a fixed input; assert probability matches to 1e-12."""
    import joblib
    import json

    pipeline = joblib.load(MODEL_PATH)

    with FIDELITY_FIXTURE.open() as f:
        input_data = json.load(f)
    with FIDELITY_EXPECTED.open() as f:
        expected = json.load(f)

    df = pd.DataFrame([input_data])
    proba = pipeline.predict_proba(df)[0, 1]

    assert abs(proba - expected["probability"]) < 1e-12, (
        f"Fidelity mismatch: got {proba:.15f}, expected {expected['probability']:.15f}"
    )
