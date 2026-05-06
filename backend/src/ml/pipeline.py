"""Build sklearn preprocessing + classifier pipelines.

All models share the same ColumnTransformer so comparison is fair.
Callers pass a classifier name string; this module returns a Pipeline
ready for fit() / predict_proba().
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build ColumnTransformer for numeric and categorical features.

    Args:
        numeric_features: Column names for numeric imputation + scaling.
        categorical_features: Column names for imputation + one-hot encoding.

    Returns:
        Unfitted ColumnTransformer.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def build_pipeline(
    model_name: str,
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int = 42,
) -> Pipeline:
    """Return an unfitted sklearn Pipeline with ColumnTransformer + classifier.

    Args:
        model_name: One of ``dummy``, ``logistic_regression``,
            ``random_forest``, ``extra_trees``.
        numeric_features: Column names for numeric preprocessing.
        categorical_features: Column names for categorical preprocessing.
        random_state: Seed for reproducibility.

    Returns:
        Unfitted sklearn Pipeline.

    Raises:
        ValueError: If model_name is not recognised.
    """
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    if model_name == "dummy":
        classifier = DummyClassifier(strategy="stratified", random_state=random_state)
    elif model_name == "logistic_regression":
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs",
        )
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_name == "extra_trees":
        classifier = ExtraTreesClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )