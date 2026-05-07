"""Alembic environment — uses Settings for DB URL, imports Base for autogenerate."""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from core.models import Base
from core.settings import get_settings

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Tables owned by MLflow — Alembic must not touch them
_MLFLOW_TABLES = {
    "experiments", "runs", "metrics", "params", "tags", "experiment_tags",
    "latest_metrics", "model_versions", "registered_models", "registered_model_tags",
    "model_version_tags", "datasets", "inputs", "input_tags", "registered_model_aliases",
    "trace_info", "trace_request_metadata", "trace_tags",
    "alembic_version",
}


def include_object(obj: object, name: str, type_: str, reflected: bool, compare_to: object) -> bool:
    if type_ == "table" and name in _MLFLOW_TABLES:
        return False
    return True


def get_url() -> str:
    return get_settings().sync_database_url


def run_migrations_offline() -> None:
    """Run migrations without a live DB connection (generates SQL only)."""
    context.configure(
        url=get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table="alembic_version_dt",  # separate from MLflow's alembic_version
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live DB connection."""
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = get_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table="alembic_version_dt",  # separate from MLflow's alembic_version
            include_object=include_object,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
