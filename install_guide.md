# Docker Workflow & Installation Guide

## Architecture Overview

The project has **two layers** of Docker:

1. **Shared infrastructure** (`~/infra/`) — runs once, shared across all projects
2. **Application services** (project's `docker-compose.yaml`) — the 4 custom services + MLflow

```
+------------------------------------------------------+
|  Shared Infrastructure (~/infra/)        [runs once]  |
|  +------------+  +---------+  +-----------+           |
|  |  Postgres  |  |  Redis  |  |  pgAdmin  |           |
|  +------------+  +---------+  +-----------+           |
|        All on network: dev_network                    |
+---------------------+--------------------------------+
                      |
+---------------------+--------------------------------+
|  Project Services (docker-compose.yaml)              |
|  +----------+  +----------+  +----------+            |
|  | service  |  |  agent   |  |  worker  |            |
|  |  :8000   |  |  :8001   |  | (no port)|            |
|  +----------+  +----------+  +----------+            |
|  +----------+  +----------+                           |
|  |  MLflow  |  |Dashboard |                          |
|  |  :5000   |  |  :8501   |                          |
|  +----------+  +----------+                           |
|        All on network: dev_network (external)         |
+------------------------------------------------------+
```

---

## Development Workflow

There are three dev modes. Pick whichever suits you.

### Mode A: Full Docker (easiest for a fresh clone)

```bash
# 1. Start shared infra (one time)
make infra-up   # equivalent: docker compose -f ~/infra/docker-compose.yml up -d

# 2. Clone, copy env, fill secrets
git clone <repo>
cd project5_drift_triage_copilot
cp .env.example .env
# Edit .env: set GOOGLE_API_KEY, POSTGRES_PASSWORD, PROMOTION_API_KEY

# 3. Train the model (downloads data + registers model in MLflow)
make data        # downloads bank-additional-full.csv
make train       # trains model, writes artifacts/

# 4. Start all project services
make up          # or: docker compose up -d
```

Everything runs in containers. Postgres/Redis are shared from `~/infra/`. The project services connect via the `dev_network` Docker network.

**Important `.env` behavior:** In `docker-compose.yaml`, the `environment:` block **overrides** `.env` for container-to-container URLs (e.g., `POSTGRES_HOST: postgres` replaces `localhost`). So even though your `.env` may say `localhost` for local dev, inside Docker the service-level variables win.

### Mode B: Hybrid local with hot reload (recommended for active development)

The `docker-compose.override.yml` (which is gitignored) mount-binds `./backend/src` into the containers and adds `--reload` to uvicorn. This means:

- You edit code locally and containers auto-restart with changes
- No rebuild needed between code changes
- The override file is **auto-merged** by Docker Compose (no special flag)

```bash
# Activate dev mode with hot reload:
cp docker-compose.override.yml.example docker-compose.override.yml
# Then restart:
docker compose up -d
```

The override replaces the production CMDs with `--reload` versions and binds `./backend/src:/app/src`.

### Mode C: Pure local (no Docker for Python services)

The Makefile has targets for running services directly on your machine:

```bash
make infra-up       # still needs Postgres + Redis in Docker
make mlflow-up      # MLflow in Docker
make dev-service    # runs uvicorn locally on :8000 with --reload
make dev-agent      # runs uvicorn locally on :8001 with --reload
make dev-worker     # runs arq worker locally
make dev-dashboard  # runs streamlit locally
```

When running locally, your `.env` uses `localhost` URLs (which is the default in `.env.example`). When running in Docker, `docker-compose.yaml` overrides them to container names (`postgres`, `redis`, etc.).

---

## Production Workflow

The production setup uses the **same `docker-compose.yaml`** but **without** the override file:

```bash
docker compose up -d
```

Key differences from dev:

- **No `docker-compose.override.yml`** — no bind mounts, no `--reload`
- Images are built fresh from the multi-stage `Dockerfile` (one file, 3 targets: `service`, `agent`, `worker`)
- Each service runs as non-root user `appuser` (UID 1001)
- `artifacts/` mounted read-only (`:ro`) for the service container
- Secrets would migrate from `.env` to **HashiCorp Vault** (see `DECISIONS.md`)
- Ports are fixed: service=8000, agent=8001, dashboard=8501, MLflow=5000

---

## Quick Start for Your Partner

```
1.  git clone <repo URL>
2.  cd project5_drift_triage_copilot
3.  cp .env.example .env
4.  Edit .env — fill in these 3 values:
    - GOOGLE_API_KEY=<your key>
    - POSTGRES_PASSWORD=<pick a password>
    - PROMOTION_API_KEY=<pick a 16+ char string>
5.  make data          # download dataset
6.  make train         # train model + register in MLflow
7.  make infra-up      # start postgres + redis (from ~/infra/)
8.  make up            # start all services

For hot-reload dev:
9.  cp docker-compose.override.yml.example docker-compose.override.yml
    # (this file is gitignored, so it won't show up in git status)
10. docker compose up -d   # now editing backend/src/ auto-reloads

Ports:
    8000  = model service API
    8001  = agent API
    8501  = Streamlit dashboard
    5000  = MLflow UI
```

### Common Gotchas

- **`make infra-up` requires `~/infra/docker-compose.yml`** to exist. Make sure the shared infra repo is checked out there. The project's `docker-compose.yaml` joins the external `dev_network` that Postgres/Redis run on. If that network doesn't exist, `docker compose up` will fail.
- **`docker-compose.override.yml` is gitignored** — it won't appear in git status and won't be committed. Each developer creates it locally.
- **`artifacts/` is gitignored** — you must run `make data` and `make train` locally. The trained model and data are not in the repo.
- **`.env` is gitignored** — never commit it. Secrets stay local.