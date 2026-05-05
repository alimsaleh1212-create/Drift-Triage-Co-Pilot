import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from pydantic import ValidationError

from src.ml.schema import BankMarketingRequest, to_api_error


MODEL_PATH = Path("artifacts/models/bank_marketing_model.joblib")


_MODEL_ARTIFACT: dict[str, Any] | None = None


def load_model_artifact() -> dict[str, Any]:
    """
    Load the saved model artifact once and reuse it.
    """
    global _MODEL_ARTIFACT

    if _MODEL_ARTIFACT is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model artifact not found at: {MODEL_PATH}")

        _MODEL_ARTIFACT = joblib.load(MODEL_PATH)

    return _MODEL_ARTIFACT


def prepare_input_row(request: BankMarketingRequest) -> pd.DataFrame:
    """
    Convert the validated Pydantic request into a one-row DataFrame
    that matches the training columns.
    """
    row = request.model_dump(by_alias=True)

    # Same feature engineering rule used during training.
    row["pdays_was_999"] = int(row["pdays"] == 999)

    return pd.DataFrame([row])


def predict_one(payload: dict) -> dict:
    """
    Validate one payload and return either:
    - prediction result
    - structured validation error
    """
    try:
        request = BankMarketingRequest.model_validate(payload)
    except ValidationError as exc:
        return to_api_error(exc)

    artifact = load_model_artifact()

    pipeline = artifact["pipeline"]
    threshold = float(artifact["threshold"])
    selected_model = artifact["selected_model"]

    row = prepare_input_row(request)

    probability_yes = float(pipeline.predict_proba(row)[0, 1])
    label = int(probability_yes >= threshold)

    return {
        "model_name": selected_model,
        "threshold_used": threshold,
        "subscribe_probability": probability_yes,
        "subscribe_label": label,
        "label_meaning": {
            "0": "no",
            "1": "yes",
        },
    }


if __name__ == "__main__":
    sample_payload = {
        "age": 35,
        "job": "admin.",
        "marital": "married",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "mon",
        "campaign": 2,
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp.var.rate": 1.1,
        "cons.price.idx": 93.994,
        "cons.conf.idx": -36.4,
        "euribor3m": 4.857,
        "nr.employed": 5191.0,
    }

    result = predict_one(sample_payload)
    print(json.dumps(result, indent=2))