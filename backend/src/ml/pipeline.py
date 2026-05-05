"""Build the sklearn preprocessing + classifier pipeline.

STUB — partner implements the body. Signatures and return types are final.
"""

from __future__ import annotations

from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]

from ml.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_pipeline(classifier: object) -> Pipeline:
    """Return a fitted-ready sklearn Pipeline with ColumnTransformer.

    Partner implements this function. Requirements per CLAUDE.md §17:
    - ColumnTransformer with impute + scale on numerics, OHE on categoricals.
    - OHE must use handle_unknown='infrequent_if_exist'.
    - All transforms are inside the Pipeline (no pre-transform leakage).

    Args:
        classifier: An sklearn-compatible estimator (e.g. RandomForestClassifier).

    Returns:
        An unfitted sklearn Pipeline ready for fit() / predict_proba().
    """
    raise NotImplementedError("ML stub — partner implements")
