"""Application settings loaded from HashiCorp Vault (secrets) and environment (AppRole creds)."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import hvac
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _fetch_vault_secrets() -> dict[str, Any]:
    """Authenticate to Vault with AppRole and return KV secrets dict."""
    addr = os.environ.get("VAULT_ADDR", "http://vault:8200")
    role_id = os.environ["VAULT_ROLE_ID"]
    secret_id = os.environ["VAULT_SECRET_ID"]

    client = hvac.Client(url=addr)
    client.auth.approle.login(role_id=role_id, secret_id=secret_id)
    response = client.secrets.kv.v1.read_secret(
        path="drift-triage",
        mount_point="secret",
    )
    return response["data"]  # type: ignore[return-value]


class Settings(BaseSettings):
    """Typed application config.

    Secrets come from Vault. Non-secret config comes from env vars or defaults.
    Only VAULT_ADDR, VAULT_ROLE_ID, VAULT_SECRET_ID come from the environment.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
    )

    # Vault AppRole — only env vars allowed
    vault_addr: str = Field("http://vault:8200", alias="VAULT_ADDR")
    vault_role_id: str = Field(..., alias="VAULT_ROLE_ID")
    vault_secret_id: str = Field(..., alias="VAULT_SECRET_ID")

    # Secrets — injected at startup from Vault; not from env
    google_api_key: str = Field(..., min_length=1)
    postgres_password: str = Field(..., min_length=1)
    promotion_api_key: str = Field(..., min_length=16)

    # Non-secret config — safe from env or defaults
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_user: str = "drift_triage"
    postgres_db: str = "drift_triage"

    redis_url: str = "redis://redis:6379"
    mlflow_tracking_uri: str = "http://mlflow:5000"

    gemini_model_cheap: str = "gemini-2.5-flash"
    gemini_model_strong: str = "gemini-2.5-pro"
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "llama3"

    # ML / drift
    random_state: int = 42
    test_size: float = 0.2
    val_size: float = 0.2
    min_recall: float = 0.75
    drift_psi_warn: float = 0.1
    drift_psi_high: float = 0.25
    drift_chi2_alpha: float = 0.05
    drift_window_size: int = 500

    # Queue
    redis_queue_name: str = "drift_actions"
    redis_max_retries: int = 3
    redis_retry_delay_base: float = 1.0

    # Services (internal)
    service_url: str = "http://service:8000"
    agent_url: str = "http://agent:8001"

    @property
    def async_database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def sync_database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance, fetching Vault secrets once."""
    vault_secrets = _fetch_vault_secrets()
    return Settings(
        google_api_key=vault_secrets["google_api_key"],
        postgres_password=vault_secrets["postgres_password"],
        promotion_api_key=vault_secrets["promotion_api_key"],
    )
