"""Root test fixtures: Settings override, async test config."""

from __future__ import annotations

import pytest

from core.settings import Settings, get_settings


@pytest.fixture(autouse=True)
def override_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject test Settings — no .env file required."""
    test_settings = Settings(
        google_api_key="test-google-api-key-for-testing",
        postgres_password="testpassword",
        promotion_api_key="test_promotion_key_16ch",
    )
    monkeypatch.setattr("core.settings.get_settings", lambda: test_settings)
    get_settings.cache_clear()