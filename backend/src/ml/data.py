"""Load and split the UCI Bank Marketing dataset.

Mandatory preprocessing per CLAUDE.md §17:
- Drop ``duration`` (post-call leakage).
- Create ``pdays_was_999`` flag (``pdays == 999`` is a sentinel).
- Keep ``unknown`` as a real category.
- Stratified 60/20/20 split with ``random_state`` from Settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import structlog

from core.settings import get_settings

log = structlog.get_logger(__name__)

TARGET_COLUMN = "y"
LEAKAGE_COLUMNS = ["duration"]


@dataclass
class DataSplit:
    """Stratified 60/20/20 train/val/test split."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    numeric_features: list[str]
    categorical_features: list[str]
    dataset_hash: str


def _sha256_of(path: Path) -> str:
    """Return hex SHA-256 digest of a file."""
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_data(csv_path: Path | None = None) -> DataSplit:
    """Load raw CSV, apply feature engineering, return stratified split.

    Args:
        csv_path: Path to bank-additional-full.csv. Defaults to
            ``artifacts/data/raw/bank-additional-full.csv``.

    Returns:
        DataSplit with train/val/test X and y, feature lists, and dataset hash.

    Raises:
        FileNotFoundError: If csv_path does not exist. Run ``make data`` first.
    """
    settings = get_settings()

    if csv_path is None:
        csv_path = Path("artifacts/data/raw/bank-additional-full.csv")

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path}. Run `make data` first.")

    log.info("data.load", path=str(csv_path))
    df = pd.read_csv(csv_path, sep=";")

    dataset_hash = _sha256_of(csv_path)

    df = df.drop(columns=LEAKAGE_COLUMNS, errors="ignore")

    #created a flag , 999 means the client was not previously contacted
    if "pdays" in df.columns:
        df["pdays_was_999"] = (df["pdays"] == 999).astype(int)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].map({"no": 0, "yes": 1})

    test_size = settings.test_size
    val_size = settings.val_size
    val_ratio = val_size / (1 - test_size)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=settings.random_state,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=val_ratio,
        stratify=y_temp,
        random_state=settings.random_state,
    )

    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

    log.info(
        "data.split",
        train=len(X_train),
        val=len(X_val),
        test=len(X_test),
        numeric=len(numeric_features),
        categorical=len(categorical_features),
    )

    return DataSplit(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        dataset_hash=dataset_hash,
    )