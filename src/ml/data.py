from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TARGET_COLUMN = "y"
LEAKAGE_COLUMNS = ["duration"]


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """
    Load the UCI Bank Marketing dataset.
    The file uses semicolon separators.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    return pd.read_csv(path, sep=";")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply project-specific cleaning rules:
    - Drop duration because it leaks the target.
    - Keep 'unknown' as a real category.
    - Add a flag for pdays == 999.
    """
    df = df.copy()

    df = df.drop(columns=LEAKAGE_COLUMNS, errors="ignore")
    #duration is only known after the phone call ends, but our model should predict before knowing the call result

    if "pdays" in df.columns:
        df["pdays_was_999"] = (df["pdays"] == 999).astype(int)
    #created a flag , 999 means the client was not previously contacted

    return df


def split_data(df: pd.DataFrame):
    """
    Create stratified 60/20/20 split:
    - 60% train
    - 20% validation
    - 20% test
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].map({"no": 0, "yes": 1})

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_feature_columns(X: pd.DataFrame):
    """
    Separate numeric and categorical features.
    """
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    return numeric_features, categorical_features


if __name__ == "__main__":
    data_path = "data/raw/bank-additional-full.csv"

    raw_df = load_raw_data(data_path)
    clean_df = clean_data(raw_df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(clean_df)
    numeric_features, categorical_features = get_feature_columns(X_train)

    print("Raw shape:", raw_df.shape)
    print("Clean shape:", clean_df.shape)

    print("\nSplit sizes:")
    print("Train:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape, y_test.shape)

    print("\nTarget distribution:")
    print("Train:")
    print(y_train.value_counts(normalize=True))
    print("Validation:")
    print(y_val.value_counts(normalize=True))
    print("Test:")
    print(y_test.value_counts(normalize=True))

    print("\nNumeric features:")
    print(numeric_features)

    print("\nCategorical features:")
    print(categorical_features)