.PHONY: install data train migrate db-create network mlflow-up dev-service dev-agent dev-worker dev-dashboard up up-full down logs test lint fmt ci demo-drift

# ── Dev setup ────────────────────────────────────────────────────────────────
install:
	cd backend && uv sync && uv run pre-commit install

# ── ML pipeline ──────────────────────────────────────────────────────────────
data:
	mkdir -p backend/artifacts/data/raw
	curl -L "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip" \
	     -o /tmp/bank.zip
	unzip -o /tmp/bank.zip -d /tmp/bank
	cp /tmp/bank/bank-additional/bank-additional-full.csv backend/artifacts/data/raw/

train:
	cd backend && ENV_FILE=../.env uv run python -m ml.train

# ── Database ──────────────────────────────────────────────────────────────────
# Create the drift_triage database on a shared postgres instance.
# Only needed when joining a pre-existing postgres (e.g. from ~/infra).
# Not needed on a fresh docker compose up — postgres container creates it automatically.
db-create:
	docker exec postgres psql -U $${POSTGRES_USER:-postgres} \
	  -c "SELECT 'CREATE DATABASE drift_triage' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'drift_triage')\gexec"

migrate:
	cd backend && uv run --no-dev alembic upgrade head

# ── Local dev (no Docker rebuild needed) ──────────────────────────────────────
mlflow-up:
	docker compose up -d mlflow

dev-service:
	cd backend && uv run uvicorn service.main:app --host 0.0.0.0 --port 8000 --reload

dev-agent:
	cd backend && uv run uvicorn agent.main:app --host 0.0.0.0 --port 8001 --reload

dev-worker:
	cd backend && uv run arq worker.main.WorkerSettings

dev-dashboard:
	cd frontend && uv run streamlit run app.py --server.port 8501

# ── Docker ───────────────────────────────────────────────────────────────────
# Ensure dev_network exists before any compose up.
# Idempotent — no-op if network already created by ~/infra or another project.
network:
	docker network create dev_network 2>/dev/null || true

# Shared infra already running (postgres/redis from ~/infra or another project).
# Starts mlflow + app services only — joins existing postgres/redis on dev_network.
up: network
	docker compose up -d

# Fresh machine — starts postgres + redis + pgadmin + all app services.
# Also use after `docker compose down -v` to recreate volumes.
up-full: network
	docker compose --profile infra up -d

down:
	docker compose down -v

logs:
	docker compose logs -f service agent worker

# ── Tests / lint ─────────────────────────────────────────────────────────────
test:
	cd backend && uv run pytest -q

lint:
	cd backend && uv run black --check src tests
	cd backend && uv run isort --check src tests
	cd backend && uv run flake8 src tests
	cd backend && uv run mypy src

fmt:
	cd backend && uv run black src tests
	cd backend && uv run isort src tests

ci: lint test

# ── Demo helpers ─────────────────────────────────────────────────────────────
demo-drift:
	cd backend && uv run python scripts/inject_drift.py \
	  --feature euribor3m --shift 2.0 --n 500
