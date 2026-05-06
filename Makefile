.PHONY: install data train infra-up infra-down mlflow-up dev-service dev-agent dev-worker dev-dashboard up down logs test lint fmt ci demo-drift

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
	cd backend && uv run python -m ml.train

# ── Migrations ─────────────────────────────────────────────────────────────────
migrate:
	cd backend && uv run --no-dev alembic upgrade head

# ── Shared infra (postgres, redis, pgadmin) ───────────────────────────────────
infra-up:
	docker compose -f ~/infra/docker-compose.yml up -d

infra-down:
	docker compose -f ~/infra/docker-compose.yml down

# ── Local dev (no Docker rebuild needed) ───────────────────────────────────────
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
up: infra-up
	docker compose up -d

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
