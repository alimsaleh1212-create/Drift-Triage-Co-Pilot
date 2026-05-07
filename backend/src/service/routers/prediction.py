"""POST /predict — run prediction and log to Postgres rolling window."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from ml.inference import predict_batch
from service.deps.classifier import get_classifier, get_threshold
from service.deps.db import get_session

router = APIRouter()
log = get_logger(__name__)


class PredictRequest(BaseModel):
    """Incoming features for a single prediction."""

    age: int = Field(..., ge=18, le=100)
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    campaign: int = Field(..., ge=1)
    pdays: int
    previous: int = Field(..., ge=0)
    poutcome: str
    emp_var_rate: float = Field(..., alias="emp.var.rate")
    cons_price_idx: float = Field(..., alias="cons.price.idx")
    cons_conf_idx: float = Field(..., alias="cons.conf.idx")
    euribor3m: float
    nr_employed: float = Field(..., alias="nr.employed")

    model_config = {"populate_by_name": True}


class PredictResponse(BaseModel):
    """Prediction result with confidence."""

    prediction_id: str
    label: int
    probability: float = Field(..., ge=0.0, le=1.0)
    threshold: float
    timestamp: datetime


async def _log_prediction(
    session: AsyncSession,
    prediction_id: str,
    features: dict[str, Any],
    label: int,
    probability: float,
) -> None:
    """Persist prediction to rolling-window table asynchronously."""
    from sqlalchemy import text

    await session.execute(
        text(
            "INSERT INTO predictions (id, features, label, probability, created_at) "
            "VALUES (:id, CAST(:features AS jsonb), :label, :probability, now())"
        ),
        {
            "id": prediction_id,
            "features": json.dumps(features),
            "label": label,
            "probability": probability,
        },
    )
    await session.commit()


def _prepare_features(payload: PredictRequest) -> dict[str, Any]:
    """Convert validated PredictRequest to a feature dict matching training columns.

    Applies the same feature engineering used during training:
    ``pdays_was_999`` flag from ``pdays == 999``.
    """
    feature_dict: dict[str, Any] = payload.model_dump(by_alias=True)
    feature_dict["pdays_was_999"] = int(feature_dict["pdays"] == 999)
    return feature_dict


@router.post("/predict", response_model=PredictResponse)
async def predict(
    payload: PredictRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    classifier: Any = Depends(get_classifier),
    threshold: float = Depends(get_threshold),
    session: AsyncSession = Depends(get_session),
) -> PredictResponse:
    """Run inference on incoming features and log to the rolling window."""
    import pandas as pd

    feature_dict = _prepare_features(payload)
    df = pd.DataFrame([feature_dict])

    proba = await asyncio.to_thread(lambda: float(predict_batch(df, classifier)[0]))
    label = int(proba >= threshold)
    prediction_id = str(uuid4())

    background_tasks.add_task(
        _log_prediction, session, prediction_id, feature_dict, label, proba
    )

    log.info(
        "prediction.complete",
        prediction_id=prediction_id,
        label=label,
        probability=round(proba, 4),
    )

    return PredictResponse(
        prediction_id=prediction_id,
        label=label,
        probability=proba,
        threshold=threshold,
        timestamp=datetime.now(timezone.utc),
    )
