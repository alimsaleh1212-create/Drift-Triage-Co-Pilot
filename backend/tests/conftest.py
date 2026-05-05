"""Root test fixtures: Settings override, async test config."""

from __future__ import annotations

import pytest

from core.settings import Settings, get_settings


@pytest.fixture(autouse=True)
def override_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass Vault — inject test Settings directly."""
    test_settings = Settings(
        vault_role_id="test-role",
        vault_secret_id="test-secret",
        google_api_key="test-google-key",
        postgres_password="testpassword",
        promotion_api_key="test_promotion_key_abc",
    )
    monkeypatch.setattr("core.settings.get_settings", lambda: test_settings)
    get_settings.cache_clear()
