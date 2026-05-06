"""Prediction helper: convert validated request to DataFrame row and predict.

Per CLAUDE.md §6, the pipeline and threshold are injected via FastAPI
Depends() from app.state — no globals, no module-level joblib.load().
This module only provides pure functions that take the pipeline and threshold
as arguments.
"""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

import structlog

from ml.schema import BankMarketingRequest, PredictResponse

log = structlog.get_logger(__name__)


def prepare_input_row(request: BankMarketingRequest) -> pd.DataFrame:
    """Convert a validated Pydantic request into a one-row DataFrame.

    Applies the same feature engineering rule used during training:
    ``pdays_was_999`` flag from ``pdays == 999``.

    Args:
        request: Validated prediction request.

    Returns:
        Single-row DataFrame with columns matching training features.
    """
    row = request.model_dump(by_alias=True)
    row["pdays_was_999"] = int(row["pdays"] == 999)
    return pd.DataFrame([row])


def predict_one(
    payload: dict,
    pipeline: Pipeline,
    threshold: float,
    model_name: str,
) -> PredictResponse | dict:
    """Validate a payload, predict, and return a structured response.

    If the payload fails Pydantic validation, returns a structured error
    dict instead of raising.

    Args:
        payload: Raw prediction request dict.
        pipeline: Fitted sklearn pipeline (injected via Depends).
        threshold: Operating threshold (injected via Depends).
        model_name: Registered model name for the response.

    Returns:
        PredictResponse on success, or dict with error details on validation failure.
    """
    from pydantic import ValidationError

    try:
        request = BankMarketingRequest.model_validate(payload)
    except ValidationError as exc:
        return {
            "error": "validation_error",
            "details": [
                {
                    "loc": list(e["loc"]),
                    "msg": e["msg"],
                    "type": e["type"],
                }
                for e in exc.errors()
            ],
        }

    row = prepare_input_row(request)
    probability_yes = float(pipeline.predict_proba(row)[0, 1])
    label = int(probability_yes >= threshold)

    return PredictResponse(
        model_name=model_name,
        threshold_used=threshold,
        subscribe_probability=probability_yes,
        subscribe_label=label,
    )