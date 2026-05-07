"""Application settings loaded from environment and .env file.

For production, migrate secrets to HashiCorp Vault — see DECISIONS.md.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application config.

    Secrets come from .env (gitignored). Non-secret config has sensible
    defaults for local development. extra="forbid" ensures typos crash
    at startup rather than silently leaving None.
    """

    model_config = SettingsConfigDict(
        env_file=[".env", "../.env"],  # works from backend/ or project root
        env_file_encoding="utf-8",
        extra="forbid",
    )

    # ── Secrets (from .env, gitignored) ──────────────────────────────
    google_api_key: str = Field(..., min_length=1)
    postgres_password: str = Field(..., min_length=1)
    promotion_api_key: str = Field(..., min_length=16)

    # ── Postgres ──────────────────────────────────────────────────────
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_user: str = "drift_triage"
    postgres_db: str = "drift_triage"

    # ── Redis ──────────────────────────────────────────────────────────
    redis_url: str = "redis://redis:6379"

    # ── MLflow ─────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = "http://mlflow:5000"

    # ── LLM — Primary: Gemini ─────────────────────────────────────────
    gemini_model: str = "gemini-2.5-flash"

    # ── LLM — Fallback: Ollama ────────────────────────────────────────
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "llama3"

    # ── ML / Drift thresholds ─────────────────────────────────────────
    random_state: int = 42
    test_size: float = 0.2
    val_size: float = 0.2
    min_recall: float = 0.75
    drift_psi_warn: float = 0.1
    drift_psi_high: float = 0.25
    drift_chi2_alpha: float = 0.05
    drift_window_size: int = 500

    # ── Queue (arq) ────────────────────────────────────────────────────
    redis_queue_name: str = "drift_actions"
    redis_max_retries: int = 3
    redis_retry_delay_base: float = 1.0

    # ── Internal service URLs ──────────────────────────────────────────
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
        # psycopg (v3) — already installed via langgraph-checkpoint-postgres
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def checkpoint_database_url(self) -> str:
        # Plain psycopg3 connection string for AsyncPostgresSaver (no SQLAlchemy prefix)
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance, reading from .env / env vars."""
    return Settings()