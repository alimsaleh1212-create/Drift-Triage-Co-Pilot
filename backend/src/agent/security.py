"""Prompt-injection prevention for all LLM inputs.

Rules per CLAUDE.md §15:
1. Sanitize external strings before format().
2. Delimit user/external content with XML tags.
3. Sanitize LLM string outputs before re-use.
4. max_output_tokens on every call (enforced in llm.py).
5. No eval() / exec() on LLM output.
6. Log suspicious patterns.
7. Pydantic is the fence.
8. Tool allowlist (enforced in graph.py).
"""

from __future__ import annotations

import re
import unicodedata

from core.logging import get_logger

log = get_logger(__name__)

_INJECTION_PATTERNS = [
    r"ignore previous",
    r"you are now",
    r"system prompt",
    r"\n\nHuman:",
    r"\n\nAssistant:",
    r"<\|system\|>",
]
_MAX_EXTERNAL_LEN = 4096


def _sanitize(text: str, max_len: int = _MAX_EXTERNAL_LEN) -> str:
    """Normalise whitespace, strip control chars, truncate, detect injections."""
    text = unicodedata.normalize("NFKC", text)
    # Strip control characters except newline/tab
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Normalise runs of whitespace
    text = re.sub(r"[ \t]+", " ", text).strip()
    text = text[:max_len]

    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            log.warning("security.injection_pattern", pattern=pattern)

    return text


def sanitize_query(text: str) -> str:
    """Sanitize a user-originated or external query string for prompt use."""
    return _sanitize(text)


def sanitize_feature_string(text: str) -> str:
    """Sanitize an LLM output string before re-use in another prompt or tool arg."""
    return _sanitize(text, max_len=512)


def delimit_external(text: str) -> str:
    """Wrap sanitized external text in XML delimiter tags."""
    return f"<external_data>\n{sanitize_query(text)}\n</external_data>"
