"""GET /drift/report — compute and cache drift report; emit webhook on severity change."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pandas as pd
from cachetools import TTLCache
from fastapi import APIRouter, BackgroundTasks, Depends, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.logging import get_logger
from core.settings import get_settings
from drift.chi2 import chi2_result
from drift.output_drift import compute_output_drift
from drift.psi import psi_result
from drift.severity import DriftReport, build_drift_report, report_to_webhook
from ml.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from ml.reference_stats import ReferenceStats
from service.deps.classifier import get_ref_stats
from service.deps.db import get_session

router = APIRouter()
log = get_logger(__name__)

_drift_cache: TTLCache[str, DriftReport] = TTLCache(maxsize=8, ttl=60)
_drift_lock = asyncio.Lock()


async def _fetch_rolling_window(
    session: AsyncSession, model_name: str, window_size: int
) -> pd.DataFrame:
    """Query the last window_size predictions from Postgres."""
    result = await session.execute(
        text(
            "SELECT features, label FROM predictions "
            "ORDER BY created_at DESC LIMIT :window"
        ),
        {"window": window_size},
    )
    rows = result.fetchall()
    if not rows:
        return pd.DataFrame()
    import json

    records = []
    for row in rows:
        features = (
            json.loads(row.features) if isinstance(row.features, str) else row.features
        )
        records.append({**features, "label": row.label})
    return pd.DataFrame(records)


async def _compute_drift(
    ref_stats: ReferenceStats,
    session: AsyncSession,
    model_name: str,
    model_version: int,
) -> DriftReport:
    """Compute a fresh DriftReport against the rolling window."""
    settings = get_settings()
    df = await _fetch_rolling_window(session, model_name, settings.drift_window_size)

    if df.empty:
        # Not enough data yet — return a low-severity placeholder
        from drift.output_drift import OutputDriftResult

        return build_drift_report(
            model_name=model_name,
            model_version=model_version,
            psi_results=[],
            chi2_results=[],
            output_drift=OutputDriftResult(
                psi=0.0,
                severity="low",
                reference_class_1_rate=ref_stats.output_proportions.get("1", 0.0),
                current_class_1_rate=0.0,
                current_n=0,
            ),
            window_size=0,
        )

    psi_results = [
        psi_result(
            feat, pd.Series(ref_stats.numeric[feat]["reference_values"]), df[feat]
        )
        for feat in NUMERIC_FEATURES
        if feat in df.columns and feat in ref_stats.numeric
    ]
    chi2_results = [
        chi2_result(feat, ref_stats.categorical[feat], df[feat])
        for feat in CATEGORICAL_FEATURES
        if feat in df.columns and feat in ref_stats.categorical
    ]
    label_col = df.get("label", pd.Series(dtype=int))
    output_drift = compute_output_drift(ref_stats.output_proportions, label_col)

    return build_drift_report(
        model_name=model_name,
        model_version=model_version,
        psi_results=psi_results,
        chi2_results=chi2_results,
        output_drift=output_drift,
        window_size=len(df),
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
    reraise=True,
)
async def _emit_webhook(client: httpx.AsyncClient, payload: dict[str, Any]) -> None:
    settings = get_settings()
    r = await client.post(f"{settings.agent_url}/webhook/drift", json=payload)
    r.raise_for_status()


async def _maybe_emit_severity_webhook(request: Request, report: DriftReport) -> None:
    """Emit webhook to agent if persisted drift severity has changed."""
    state_key = f"{report.model_name}:{report.model_version}"

    async with request.app.state.SessionLocal() as session:
        result = await session.execute(
            text(
                "SELECT last_severity, last_report_id "
                "FROM drift_alert_state "
                "WHERE key = :key"
            ),
            {"key": state_key},
        )
        row = result.fetchone()
        previous = row.last_severity if row else None
        if previous == report.severity:
            return

        payload = report_to_webhook(report, previous_severity=previous)
        try:
            await _emit_webhook(
                request.app.state.http_client, payload.model_dump(mode="json")
            )
        except Exception as exc:
            await session.rollback()
            log.warning(
                "drift.webhook.failed",
                error=str(exc),
                report_id=report.report_id,
                severity=report.severity,
                previous=previous,
            )
            return

        await session.execute(
            text(
                "INSERT INTO drift_alert_state "
                "(key, last_severity, last_report_id, updated_at) "
                "VALUES (:key, :severity, :report_id, now()) "
                "ON CONFLICT (key) DO UPDATE SET "
                "last_severity = EXCLUDED.last_severity, "
                "last_report_id = EXCLUDED.last_report_id, "
                "updated_at = now()"
            ),
            {
                "key": state_key,
                "severity": report.severity,
                "report_id": report.report_id,
            },
        )
        await session.commit()
        log.info(
            "drift.webhook.sent",
            report_id=report.report_id,
            severity=report.severity,
            previous=previous,
        )


@router.get("/drift/report", response_model=DriftReport)
async def get_drift_report(
    background_tasks: BackgroundTasks,
    request: Request,
    ref_stats: ReferenceStats = Depends(get_ref_stats),
    session: AsyncSession = Depends(get_session),
) -> DriftReport:
    """Return current drift report; computed at most once per 60 seconds (TTL cache)."""
    model_name: str = request.app.state.model_name
    key = model_name

    if key in _drift_cache:
        return _drift_cache[key]

    async with _drift_lock:
        if key in _drift_cache:
            return _drift_cache[key]

        report = await _compute_drift(ref_stats, session, model_name, model_version=1)
        _drift_cache[key] = report

    background_tasks.add_task(_maybe_emit_severity_webhook, request, report)
    return report
