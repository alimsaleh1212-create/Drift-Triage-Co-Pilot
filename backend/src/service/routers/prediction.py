"""POST /predict — run prediction and log to Postgres rolling window."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.settings import get_settings
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
    label: int  # 0 or 1
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
            "VALUES (:id, :features::jsonb, :label, :probability, now())"
        ),
        {
            "id": prediction_id,
            "features": str(features),
            "label": label,
            "probability": probability,
        },
    )
    await session.commit()


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
    feature_dict = payload.model_dump(by_alias=True)
    # Engineer was_previously_contacted inside the request path
    feature_dict["was_previously_contacted"] = int(feature_dict["pdays"] != 999)

    df = pd.DataFrame([feature_dict])

    # sklearn is CPU-bound — wrap to avoid blocking the event loop
    proba = await asyncio.to_thread(
        lambda: classifier.predict_proba(df)[0, 1]
    )
    label = int(proba >= threshold)
    prediction_id = str(uuid4())

    background_tasks.add_task(
        _log_prediction, session, prediction_id, feature_dict, label, float(proba)
    )

    log.info(
        "prediction.complete",
        prediction_id=prediction_id,
        label=label,
        probability=round(float(proba), 4),
    )

    return PredictResponse(
        prediction_id=prediction_id,
        label=label,
        probability=float(proba),
        threshold=threshold,
        timestamp=datetime.now(timezone.utc),
    )
