"""Inject a generated demo batch through the real prediction API.

This script never writes directly to Postgres. Each row is converted to the
existing prediction request schema and sent to POST /api/v1/predict.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH_DIR = PROJECT_ROOT / "data/demo_batches"
DEFAULT_SOURCE_CANDIDATES = [
    PROJECT_ROOT / "data/raw/bank-additional-full.csv",
    PROJECT_ROOT / "backend/artifacts/data/raw/bank-additional-full.csv",
]

SCENARIO_EXPECTED_ACTION = {
    "normal": "no_action / monitor",
    "replay_drift": "replay_test",
    "retrain_drift": "retrain",
    "rollback_drift": "rollback or another production-touching HIL action",
}

PREDICTION_FIELDS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]

CATEGORICAL_FIELDS = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]


def default_service_url() -> str:
    """Resolve service URL from env vars or a local default."""
    return (
        os.getenv("MODEL_SERVICE_URL")
        or os.getenv("SERVICE_URL")
        or os.getenv("API_BASE_URL")
        or "http://localhost:8000"
    ).rstrip("/")


def batch_path_for(scenario: str, n: int, batch_dir: Path = DEFAULT_BATCH_DIR) -> Path:
    """Return the generated batch path, falling back to the standard 2000 file."""
    exact = batch_dir / f"{scenario}_{n}.csv"
    if exact.exists():
        return exact
    return batch_dir / f"{scenario}_2000.csv"


def load_allowed_categories(source_csv: Path | None = None) -> dict[str, set[str]]:
    """Load valid categorical values from the original dataset."""
    path = source_csv or _find_source_csv()
    source = pd.read_csv(path, sep=";")
    return {
        column: set(source[column].dropna().astype(str).unique())
        for column in CATEGORICAL_FIELDS
        if column in source.columns
    }


def build_prediction_payload(
    row: pd.Series | dict[str, Any],
    allowed_categories: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    """Build the exact payload accepted by the existing prediction endpoint."""
    raw = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    payload: dict[str, Any] = {}

    for field in PREDICTION_FIELDS:
        if field not in raw:
            raise ValueError(f"Missing prediction field: {field}")
        value = raw[field]
        if pd.isna(value):
            raise ValueError(f"Null prediction field: {field}")
        if field in CATEGORICAL_FIELDS:
            value = str(value)
            if allowed_categories is not None and value not in allowed_categories.get(
                field, set()
            ):
                raise ValueError(f"Invalid category for {field}: {value}")
        else:
            value = _json_number(value, field)
        payload[field] = value

    extra = set(payload) - set(PREDICTION_FIELDS)
    if extra:
        raise ValueError(f"Unexpected payload fields: {sorted(extra)}")
    if "y" in payload or "duration" in payload:
        raise ValueError("Payload must not include y or duration")
    return payload


def inject_batch(
    scenario: str,
    n: int,
    service_url: str,
    no_drift_report: bool = False,
    timeout_seconds: float = 10.0,
    progress_every: int = 200,
) -> int:
    """Send a batch through /predict and print a drift summary."""
    path = batch_path_for(scenario, n)
    if not path.exists():
        raise FileNotFoundError(
            f"Demo batch not found: {path}. Run scripts/generate_demo_batches.py first."
        )

    df = pd.read_csv(path).head(n)
    allowed_categories = load_allowed_categories()
    predict_url = f"{service_url.rstrip('/')}/api/v1/predict"
    drift_url = f"{service_url.rstrip('/')}/api/v1/drift/report"

    success_count = 0
    failure_count = 0
    error_examples: list[dict[str, Any]] = []
    started = time.time()

    for idx, row in df.iterrows():
        try:
            payload = build_prediction_payload(row, allowed_categories)
            _post_json(predict_url, payload, timeout_seconds)
            success_count += 1
        except Exception as exc:
            failure_count += 1
            if len(error_examples) < 5:
                error_examples.append({"row": int(idx), "error": str(exc)})

        current = int(idx) + 1
        if current % progress_every == 0 or current == len(df):
            print(f"progress: {current}/{len(df)} rows sent")

    print()
    print(f"scenario: {scenario}")
    print(f"rows requested: {len(df)}")
    print(f"prediction success count: {success_count}")
    print(f"prediction failure count: {failure_count}")
    print(f"elapsed seconds: {time.time() - started:.1f}")
    print(f"expected action: {SCENARIO_EXPECTED_ACTION.get(scenario, 'unknown')}")

    if error_examples:
        print("error examples:")
        for example in error_examples:
            print(f"  row {example['row']}: {example['error']}")

    if success_count == 0 and len(df) > 0:
        print("all prediction requests failed; skipping drift report")
        return 2

    if not no_drift_report:
        try:
            report = _get_json(drift_url, timeout_seconds)
            print_drift_summary(report)
        except Exception as exc:
            print(f"drift report failed: {exc}")
            return 1

    return 0 if failure_count == 0 else 1


def print_drift_summary(report: dict[str, Any]) -> None:
    """Print a compact, readable drift report summary."""
    print()
    print(f"drift severity: {report.get('severity', 'unknown')}")
    print(f"window size: {report.get('window_size', 'unknown')}")
    print("webhook emitted: not reported by drift endpoint")

    psi_results = report.get("psi_results") or []
    if psi_results:
        top_psi = sorted(psi_results, key=lambda r: r.get("psi", 0), reverse=True)[:5]
        print("PSI summary:")
        for item in top_psi:
            print(
                f"  {item.get('feature')}: psi={item.get('psi', 0):.4f}, "
                f"severity={item.get('severity')}"
            )
    else:
        print("PSI summary: none returned")

    chi2_results = report.get("chi2_results") or []
    if chi2_results:
        top_chi2 = sorted(chi2_results, key=lambda r: r.get("p_value", 1))[:5]
        print("chi-square summary:")
        for item in top_chi2:
            print(
                f"  {item.get('feature')}: p={item.get('p_value', 1):.4g}, "
                f"severity={item.get('severity')}"
            )
    else:
        print("chi-square summary: none returned")

    output = report.get("output_drift") or {}
    if output:
        print(
            "output drift summary: "
            f"psi={output.get('psi', 0):.4f}, "
            f"severity={output.get('severity')}, "
            f"reference_class_1_rate={output.get('reference_class_1_rate', 0):.4f}, "
            f"current_class_1_rate={output.get('current_class_1_rate', 0):.4f}"
        )
    else:
        print("output drift summary: none returned")


def _post_json(
    url: str, payload: dict[str, Any], timeout_seconds: float
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return _send_json_request(request, timeout_seconds)


def _get_json(url: str, timeout_seconds: float) -> dict[str, Any]:
    request = urllib.request.Request(url, method="GET")
    return _send_json_request(request, timeout_seconds)


def _send_json_request(
    request: urllib.request.Request,
    timeout_seconds: float,
) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def _json_number(value: Any, field: str) -> int | float:
    """Convert pandas/numpy scalar values to JSON-serializable numbers."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = value
    else:
        number = float(value)

    if field in {"age", "campaign", "pdays", "previous"}:
        return int(number)
    return float(number)


def _find_source_csv() -> Path:
    for candidate in DEFAULT_SOURCE_CANDIDATES:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(p) for p in DEFAULT_SOURCE_CANDIDATES)
    raise FileNotFoundError(f"bank-additional-full.csv not found. Searched: {searched}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inject demo batch via prediction API."
    )
    parser.add_argument("scenario", choices=sorted(SCENARIO_EXPECTED_ACTION))
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--service-url", default=default_service_url())
    parser.add_argument("--no-drift-report", action="store_true")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    status = inject_batch(
        scenario=args.scenario,
        n=args.n,
        service_url=args.service_url,
        no_drift_report=args.no_drift_report,
        timeout_seconds=args.timeout,
    )
    sys.exit(status)


if __name__ == "__main__":
    main()
