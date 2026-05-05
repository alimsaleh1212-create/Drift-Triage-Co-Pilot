"""Tool: fetch the current drift report from the model service."""

from __future__ import annotations

import httpx
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from agent.tools.base import BaseTool
from core.settings import get_settings
from drift.severity import DriftReport


class InspectDriftInput(BaseModel):
    model_name: str = Field(..., min_length=1)


class InspectDriftOutput(BaseModel):
    report: DriftReport


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
    reraise=True,
)
async def _fetch_report(model_name: str) -> DriftReport:
    settings = get_settings()
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{settings.service_url}/api/v1/drift/report")
        r.raise_for_status()
    return DriftReport.model_validate(r.json())


class InspectDrift(BaseTool[InspectDriftInput, InspectDriftOutput]):
    """Fetch the latest drift report from the model service."""

    name = "inspect_drift"
    input_schema = InspectDriftInput
    output_schema = InspectDriftOutput

    async def run(self, args: InspectDriftInput) -> InspectDriftOutput:
        report = await _fetch_report(args.model_name)
        return InspectDriftOutput(report=report)
