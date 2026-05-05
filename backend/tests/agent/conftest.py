"""FakeLLM + snapshot fixtures for agent trajectory tests.

FakeLLM replays pre-recorded responses in order — no API key required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

SNAPSHOTS_DIR = Path(__file__).parent / "snapshots"


class FakeLLM:
    """Replays recorded LLM responses from a snapshot scenario."""

    def __init__(self, llm_responses: list[dict[str, Any]]) -> None:
        self._responses = iter(llm_responses)

    async def __call__(self, prompt: str, schema: type, **kwargs: Any) -> Any:
        response_data = next(self._responses)
        return schema.model_validate(response_data["args"])


def load_snapshot(scenario: str) -> dict[str, Any]:
    path = SNAPSHOTS_DIR / f"{scenario}.json"
    with path.open() as f:
        return json.load(f)


@pytest.fixture()
def fake_llm_factory():
    """Return a factory that builds FakeLLM from a snapshot scenario name."""
    def _factory(scenario: str) -> FakeLLM:
        snapshot = load_snapshot(scenario)
        return FakeLLM(snapshot["llm_responses"])
    return _factory
