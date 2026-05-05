"""Load and split the UCI Bank Marketing dataset.

STUB — partner implements the body. Signatures and return types are final;
downstream code (service, drift) imports from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]

from core.settings import get_settings

RAW_CSV = Path("artifacts/data/raw/bank-additional-full.csv")

# Features kept after removing the leaky `duration` column
NUMERIC_FEATURES: list[str] = [
    "age",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
    "was_previously_contacted",  # engineered from pdays==999
]

CATEGORICAL_FEATURES: list[str] = [
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

TARGET: str = "y"


@dataclass
class DataSplit:
    """Stratified 60/20/20 train/val/test split."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    dataset_hash: str


def load_data(csv_path: Path = RAW_CSV) -> DataSplit:
    """Load raw CSV, apply feature engineering, return stratified split.

    Partner implements this function. The following transformations are
    mandatory per CLAUDE.md §17:
    - Drop 'duration' (post-call leakage).
    - Create 'was_previously_contacted' flag (pdays != 999).
    - Keep 'unknown' as a real category.
    - Stratified 60/20/20 split with random_state=42.

    Args:
        csv_path: Path to bank-additional-full.csv.

    Returns:
        DataSplit with train/val/test X and y, plus SHA-256 dataset hash.

    Raises:
        FileNotFoundError: If csv_path does not exist. Run `make data` first.
        ValueError: If required columns are missing.
    """
    raise NotImplementedError("ML stub — partner implements")
