.PHONY: install data train up down logs test lint fmt ci demo-drift

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

# ── Docker ───────────────────────────────────────────────────────────────────
up:
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
