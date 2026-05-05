"""LLM client factory: Gemini primary with Ollama fallback.

Per CLAUDE.md §14:
- Gemini (gemini-2.5-flash) is the primary LLM for all calls.
- Ollama (local, e.g. llama3) is the fallback if Gemini is unavailable.
- Fallback is automatic: on TimeoutException / NetworkError / 5xx from Gemini,
  retry once, then fall back to Ollama.
- @lru_cache(maxsize=1) per client.
- max_output_tokens on every call.
- Log which provider served each call.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.logging import get_logger
from core.settings import get_settings

log = get_logger(__name__)

M = TypeVar("M", bound=BaseModel)

_MAX_OUTPUT_TOKENS = 2048


@lru_cache(maxsize=1)
def _get_gemini_client() -> Any:
    import google.generativeai as genai

    settings = get_settings()
    genai.configure(api_key=settings.google_api_key)
    return genai.GenerativeModel(settings.gemini_model)


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
    reraise=True,
)
async def _call_gemini(prompt: str, schema: type[M]) -> M:
    """Call Gemini with JSON response schema and return parsed Pydantic model."""
    import google.generativeai as genai

    model = _get_gemini_client()
    response = await model.generate_content_async(
        prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            max_output_tokens=_MAX_OUTPUT_TOKENS,
        ),
    )
    return schema.model_validate_json(response.text)


async def _call_ollama(prompt: str, schema: type[M]) -> M:
    """Call local Ollama as fallback and return parsed Pydantic model."""
    settings = get_settings()
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            f"{settings.ollama_base_url}/api/generate",
            json={
                "model": settings.ollama_model,
                "prompt": prompt + "\n\nRespond with valid JSON only.",
                "stream": False,
            },
        )
        r.raise_for_status()
    text = r.json()["response"]
    return schema.model_validate_json(text)


async def call_llm(prompt: str, schema: type[M]) -> M:
    """Call LLM with structured JSON output, falling back to Ollama.

    Args:
        prompt: Full prompt string (system + user already concatenated).
        schema: Pydantic model class for response validation.

    Returns:
        Validated Pydantic model instance.
    """
    try:
        result = await _call_gemini(prompt, schema)
        log.info("llm.call", provider="gemini", schema=schema.__name__)
        return result
    except Exception as exc:
        log.warning("llm.fallback", provider="gemini", error=str(exc))
        result = await _call_ollama(prompt, schema)
        log.info("llm.call", provider="ollama", schema=schema.__name__)
        return result