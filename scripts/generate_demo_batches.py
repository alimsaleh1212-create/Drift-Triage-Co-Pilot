"""Generate Friday demo batches from the real UCI Bank Marketing CSV.

The output files keep the original columns, including ``duration`` and ``y``.
The injector removes columns that are not accepted by the prediction API.

Scenarios:
- normal: unmodified real rows, expected low drift.
- replay_drift: moderate macro/categorical shift, expected replay_test.
- retrain_drift: stronger feature drift, expected retrain.
- rollback_drift: severe abnormal/output shift, expected HIL production action.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_CANDIDATES = [
    PROJECT_ROOT / "data/raw/bank-additional-full.csv",
    PROJECT_ROOT / "backend/artifacts/data/raw/bank-additional-full.csv",
]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/demo_batches"

SCENARIOS = ("normal", "replay_drift", "retrain_drift", "rollback_drift")
CATEGORICAL_COLUMNS = [
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


def find_source_csv(explicit_path: str | None = None) -> Path:
    """Return the source CSV path, preferring an explicit path if provided."""
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Source CSV not found: {path}")
        return path

    for candidate in DEFAULT_SOURCE_CANDIDATES:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(p) for p in DEFAULT_SOURCE_CANDIDATES)
    raise FileNotFoundError(f"bank-additional-full.csv not found. Searched: {searched}")


def load_source(csv_path: Path) -> pd.DataFrame:
    """Load the real UCI Bank Marketing CSV."""
    return pd.read_csv(csv_path, sep=";")


def generate_batches(
    source_df: pd.DataFrame,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    rows_per_batch: int = 2000,
    random_state: int = 42,
) -> dict[str, Path]:
    """Generate all scenario CSVs and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for offset, scenario in enumerate(SCENARIOS):
        batch = _sample_scenario_base(
            source_df, scenario, rows_per_batch, random_state + offset
        )
        batch = apply_scenario_mutations(
            batch, source_df, scenario, random_state + offset
        )
        path = output_dir / f"{scenario}_{rows_per_batch}.csv"
        batch.to_csv(path, index=False)
        paths[scenario] = path
    return paths


def _sample_scenario_base(
    source_df: pd.DataFrame,
    scenario: str,
    rows_per_batch: int,
    random_state: int,
) -> pd.DataFrame:
    """Sample real rows; severe rollback starts from more positive rows."""
    replace = len(source_df) < rows_per_batch

    if scenario == "rollback_drift" and "y" in source_df.columns:
        positives = source_df[source_df["y"] == "yes"]
        if len(positives) >= max(50, rows_per_batch // 4):
            return positives.sample(
                n=rows_per_batch,
                replace=len(positives) < rows_per_batch,
                random_state=random_state,
            ).reset_index(drop=True)

    return source_df.sample(
        n=rows_per_batch,
        replace=replace,
        random_state=random_state,
    ).reset_index(drop=True)


def apply_scenario_mutations(
    batch: pd.DataFrame,
    source_df: pd.DataFrame,
    scenario: str,
    random_state: int = 42,
) -> pd.DataFrame:
    """Apply drift mutations while keeping values valid for the source dataset."""
    rng = np.random.default_rng(random_state)
    mutated = batch.copy()

    if scenario == "normal":
        return mutated

    if scenario == "replay_drift":
        # Mild macro shift plus moderate campaign-channel seasonality.
        _shift_numeric(mutated, source_df, "euribor3m", 0.35)
        _shift_numeric(mutated, source_df, "cons.price.idx", 0.08)
        _set_fraction(mutated, "contact", "telephone", 0.35, rng, source_df)
        _set_fraction(mutated, "month", "nov", 0.25, rng, source_df)
        _set_fraction(mutated, "job", "technician", 0.20, rng, source_df)
        return mutated

    if scenario == "retrain_drift":
        # Strong economic-regime shift plus clear categorical distribution changes.
        _shift_numeric(mutated, source_df, "euribor3m", 1.25)
        _shift_numeric(mutated, source_df, "emp.var.rate", 1.20)
        _shift_numeric(mutated, source_df, "cons.price.idx", 0.35)
        _shift_numeric(mutated, source_df, "nr.employed", 90.0)
        _set_fraction(mutated, "contact", "telephone", 0.70, rng, source_df)
        _set_fraction(mutated, "month", "may", 0.65, rng, source_df)
        _set_fraction(mutated, "poutcome", "nonexistent", 0.80, rng, source_df)
        _set_fraction(mutated, "job", "blue-collar", 0.45, rng, source_df)
        return mutated

    if scenario == "rollback_drift":
        # Severe abnormal traffic: rows resemble a narrow, high-score population.
        # All values are still real categories/ranges from the original CSV.
        _set_numeric_quantile(mutated, source_df, "euribor3m", 0.02)
        _set_numeric_quantile(mutated, source_df, "emp.var.rate", 0.02)
        _set_numeric_quantile(mutated, source_df, "nr.employed", 0.02)
        _set_numeric_quantile(mutated, source_df, "cons.price.idx", 0.98)
        _set_numeric_quantile(mutated, source_df, "previous", 0.98)
        _set_numeric_quantile(mutated, source_df, "campaign", 0.05)
        _set_low_pdays(mutated, source_df)
        _set_fraction(mutated, "contact", "cellular", 0.95, rng, source_df)
        _set_fraction(mutated, "month", "mar", 0.45, rng, source_df)
        _set_fraction(mutated, "month", "oct", 0.30, rng, source_df)
        _set_fraction(mutated, "poutcome", "success", 0.80, rng, source_df)
        _set_fraction(mutated, "job", "retired", 0.50, rng, source_df)
        _set_fraction(mutated, "job", "student", 0.25, rng, source_df)
        return mutated

    raise ValueError(f"Unknown scenario: {scenario}")


def validate_no_invalid_categories(
    batch: pd.DataFrame, source_df: pd.DataFrame
) -> None:
    """Raise if a generated batch contains categories absent from the source CSV."""
    for column in CATEGORICAL_COLUMNS:
        if column not in batch.columns or column not in source_df.columns:
            continue
        allowed = set(source_df[column].dropna().astype(str).unique())
        observed = set(batch[column].dropna().astype(str).unique())
        invalid = observed - allowed
        if invalid:
            raise ValueError(f"{column} has invalid categories: {sorted(invalid)}")


def _set_fraction(
    df: pd.DataFrame,
    column: str,
    value: str,
    fraction: float,
    rng: np.random.Generator,
    source_df: pd.DataFrame,
) -> None:
    """Set a fraction of rows to a category only if it exists in the source."""
    if column not in df.columns or column not in source_df.columns:
        return
    if value not in set(source_df[column].dropna().astype(str).unique()):
        return

    n = max(1, int(len(df) * fraction))
    idx = rng.choice(df.index.to_numpy(), size=min(n, len(df)), replace=False)
    df.loc[idx, column] = value


def _shift_numeric(
    df: pd.DataFrame, source_df: pd.DataFrame, column: str, delta: float
) -> None:
    """Shift a numeric column and clip to the original observed range."""
    if column not in df.columns or column not in source_df.columns:
        return
    values = pd.to_numeric(df[column], errors="coerce") + delta
    source_values = pd.to_numeric(source_df[column], errors="coerce")
    df[column] = values.clip(source_values.min(), source_values.max())


def _set_numeric_quantile(
    df: pd.DataFrame,
    source_df: pd.DataFrame,
    column: str,
    quantile: float,
) -> None:
    """Set a numeric column near a source quantile to create distribution shift."""
    if column not in df.columns or column not in source_df.columns:
        return
    source_values = pd.to_numeric(source_df[column], errors="coerce")
    value = float(source_values.quantile(quantile))
    df[column] = value


def _set_low_pdays(df: pd.DataFrame, source_df: pd.DataFrame) -> None:
    """Move pdays away from the 999 sentinel while using observed source values."""
    if "pdays" not in df.columns or "pdays" not in source_df.columns:
        return
    observed = pd.to_numeric(source_df["pdays"], errors="coerce")
    non_sentinel = observed[(observed.notna()) & (observed < 999)]
    df["pdays"] = int(non_sentinel.quantile(0.25)) if not non_sentinel.empty else 999


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo drift batches.")
    parser.add_argument(
        "--source-csv", default=None, help="Path to bank-additional-full.csv"
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    source_path = find_source_csv(args.source_csv)
    source_df = load_source(source_path)
    paths = generate_batches(
        source_df=source_df,
        output_dir=Path(args.output_dir),
        rows_per_batch=args.rows,
        random_state=args.random_state,
    )

    for scenario, path in paths.items():
        batch = pd.read_csv(path)
        validate_no_invalid_categories(batch, source_df)
        print(f"{scenario}: wrote {len(batch)} rows to {path}")


if __name__ == "__main__":
    main()
