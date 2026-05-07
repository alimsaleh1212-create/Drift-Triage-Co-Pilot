# syntax=docker/dockerfile:1
# Multi-stage build for backend Python services.
# Target stage selects which entrypoint to use:
#   --target service   → uvicorn drift_triage.service.main:app
#   --target agent     → uvicorn drift_triage.agent.main:app
#   --target worker    → arq drift_triage.worker.main.WorkerSettings

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_HTTP_TIMEOUT=300 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libpq5 && rm -rf /var/lib/apt/lists/*

RUN pip install uv==0.6.14

COPY pyproject.toml .
COPY src/ src/

RUN uv sync --no-dev

# ── migrations ─────────────────────────────────────────────────────────────
FROM base AS migrate

COPY alembic.ini .
COPY migrations/ migrations/

CMD ["uv", "run", "--no-dev", "alembic", "upgrade", "head"]

# ── service ────────────────────────────────────────────────────────────────
FROM base AS service

RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uv", "run", "--no-dev", "uvicorn", "service.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]

# ── agent ──────────────────────────────────────────────────────────────────
FROM base AS agent

RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8001
CMD ["uv", "run", "--no-dev", "uvicorn", "agent.main:app", \
     "--host", "0.0.0.0", "--port", "8001"]

# ── worker ─────────────────────────────────────────────────────────────────
FROM base AS worker

RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

CMD ["uv", "run", "--no-dev", "arq", "worker.main.WorkerSettings"]
