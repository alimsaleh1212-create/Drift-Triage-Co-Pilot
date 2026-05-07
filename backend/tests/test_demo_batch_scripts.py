"""Offline tests for demo batch generation and API payload shaping."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENERATE_SCRIPT = PROJECT_ROOT / "scripts/generate_demo_batches.py"
INJECT_SCRIPT = PROJECT_ROOT / "scripts/inject_demo_batch.py"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generate_demo_batches = _load_module("generate_demo_batches", GENERATE_SCRIPT)
inject_demo_batch = _load_module("inject_demo_batch", INJECT_SCRIPT)

CATEGORICAL_COLUMNS = generate_demo_batches.CATEGORICAL_COLUMNS
SCENARIOS = generate_demo_batches.SCENARIOS
PREDICTION_FIELDS = inject_demo_batch.PREDICTION_FIELDS
build_prediction_payload = inject_demo_batch.build_prediction_payload
generate_batches = generate_demo_batches.generate_batches
validate_no_invalid_categories = generate_demo_batches.validate_no_invalid_categories


def test_generated_batches_have_2000_rows(tmp_path: Path) -> None:
    source = _make_source_df(2500)

    paths = generate_batches(source, output_dir=tmp_path, rows_per_batch=2000)

    assert set(paths) == set(SCENARIOS)
    for path in paths.values():
        assert len(pd.read_csv(path)) == 2000


def test_generated_batches_do_not_introduce_invalid_categories(tmp_path: Path) -> None:
    source = _make_source_df(2500)

    paths = generate_batches(source, output_dir=tmp_path, rows_per_batch=2000)

    for path in paths.values():
        batch = pd.read_csv(path)
        validate_no_invalid_categories(batch, source)
        for column in CATEGORICAL_COLUMNS:
            assert set(batch[column].unique()).issubset(set(source[column].unique()))


def test_prediction_payload_builder_excludes_target_and_duration() -> None:
    source = _make_source_df(1)
    row = source.iloc[0]
    allowed = {
        column: set(source[column].astype(str).unique())
        for column in CATEGORICAL_COLUMNS
    }

    payload = build_prediction_payload(row, allowed)

    assert "y" not in payload
    assert "duration" not in payload
    assert set(payload) == set(PREDICTION_FIELDS)


def test_prediction_payload_contains_expected_schema_fields() -> None:
    source = _make_source_df(1)
    row = source.iloc[0]

    payload = build_prediction_payload(row)

    assert list(payload.keys()) == PREDICTION_FIELDS
    assert isinstance(payload["age"], int)
    assert isinstance(payload["campaign"], int)
    assert isinstance(payload["emp.var.rate"], float)
    assert isinstance(payload["nr.employed"], float)


def _make_source_df(n: int) -> pd.DataFrame:
    jobs = [
        "admin.",
        "blue-collar",
        "technician",
        "retired",
        "student",
    ]
    months = ["mar", "may", "nov", "oct", "sep"]
    poutcomes = ["failure", "nonexistent", "success"]
    contacts = ["cellular", "telephone"]

    rows = []
    for i in range(n):
        rows.append(
            {
                "age": 18 + (i % 60),
                "job": jobs[i % len(jobs)],
                "marital": ["married", "single", "divorced"][i % 3],
                "education": ["university.degree", "high.school", "basic.9y"][i % 3],
                "default": ["no", "unknown"][i % 2],
                "housing": ["yes", "no", "unknown"][i % 3],
                "loan": ["no", "yes", "unknown"][i % 3],
                "contact": contacts[i % len(contacts)],
                "month": months[i % len(months)],
                "day_of_week": ["mon", "tue", "wed", "thu", "fri"][i % 5],
                "duration": 100 + (i % 400),
                "campaign": 1 + (i % 10),
                "pdays": 999 if i % 4 else 3 + (i % 20),
                "previous": i % 6,
                "poutcome": poutcomes[i % len(poutcomes)],
                "emp.var.rate": -3.4 + (i % 50) * 0.1,
                "cons.price.idx": 92.0 + (i % 40) * 0.05,
                "cons.conf.idx": -50.0 + (i % 30) * 0.4,
                "euribor3m": 0.6 + (i % 50) * 0.08,
                "nr.employed": 5000.0 + (i % 80) * 3.0,
                "y": "yes" if i % 9 == 0 else "no",
            }
        )
    return pd.DataFrame(rows)
