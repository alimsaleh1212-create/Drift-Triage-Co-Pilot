# Runbook — Drift Triage Co-Pilot

## Common Operations

### Demo batch injection

Generate four Friday demo batches from the original UCI Bank Marketing
`bank-additional-full.csv`:

```bash
python scripts/generate_demo_batches.py
```

The generated files are written to:

- `data/demo_batches/normal_2000.csv`
- `data/demo_batches/replay_drift_2000.csv`
- `data/demo_batches/retrain_drift_2000.csv`
- `data/demo_batches/rollback_drift_2000.csv`

Inject a batch through the real prediction API:

```bash
python scripts/inject_demo_batch.py normal --n 2000
python scripts/inject_demo_batch.py replay_drift --n 2000
python scripts/inject_demo_batch.py retrain_drift --n 2000
python scripts/inject_demo_batch.py rollback_drift --n 2000
```

Use `--service-url http://localhost:8000` when the service is running locally,
or set `MODEL_SERVICE_URL`, `SERVICE_URL`, or `API_BASE_URL`.

Expected demo behavior:

- `normal`: low/no drift, monitor/no action.
- `replay_drift`: mild/moderate drift, expected `replay_test`.
- `retrain_drift`: strong feature drift, expected `retrain`.
- `rollback_drift`: severe abnormal/output shift, expected production-touching
  action requiring HIL approval.

The data comes from the original CSV, not from an LLM. The injector removes
`y` and `duration` from prediction requests and sends rows to
`POST /api/v1/predict`; it does not insert directly into Postgres.

For reliable scenario isolation, run each scenario on a clean database or set
the service drift window to the injected batch size. The current schema does
not tag demo predictions, so the demo scripts intentionally avoid deleting or
mutating database rows.

### Full restart (clean slate)
```bash
docker compose down -v      # removes volumes
make train                  # re-register model
docker compose up -d
```

### View live logs
```bash
docker compose logs -f service agent worker
```

### Inject drift for demo
```bash
make demo-drift             # shifts euribor3m +2σ in rolling-window table
```

### Inspect DLQ
```bash
docker compose exec redis redis-cli lrange drift_actions:dlq 0 -1
```

### Regen .env secrets (if rotated)
```bash
# Edit .env directly — secrets are stored there, fetched by Settings at startup
# After updating .env, restart services:
docker compose restart service agent worker
```

---

## Failure Modes

### `service` won't start — "No Production model found"
**Cause:** MLflow has no Production model version.
**Fix:** `make train`. Then in MLflow UI, promote the registered model from Staging to Production, or wait for `register.py` to auto-promote.

### `agent` crash on startup — "checkpoint store setup failed"
**Cause:** Postgres not ready or LangGraph tables missing.
**Fix:** Confirm `postgres` healthcheck is green: `docker compose ps`. Then: `docker compose restart agent`.

### Webhook not reaching agent — "drift.webhook.failed"
**Cause:** Agent container not healthy when severity changed.
**Fix:** `docker compose restart agent`, then trigger drift again: `make demo-drift`.

### arq job stuck in queue
**Check:** `docker compose logs worker`. If worker is running but jobs don't execute: check Redis connectivity with `docker compose exec redis redis-cli ping`.
**Fix:** `docker compose restart worker`.

### HIL approval button unresponsive
**Cause:** Agent container restarted and the investigation thread resumed from checkpoint but the dashboard still shows old state.
**Fix:** Refresh dashboard. If still stuck, check `docker compose logs agent` for graph errors.

### LLM API down (Gemini unavailable)
**Symptom:** Agent logs show `llm.fallback provider=gemini`. Ollama fallback activates automatically.
**Check Ollama is up:** The Dockerfile uses Ollama as a sidecar — confirm `ollama` service is healthy if you added it, or add it to `docker-compose.yaml`.

### Model artifact gone from MLflow (URI 404)
**Cause:** MLflow artifact storage (`mlflow_data` volume) was removed.
**Fix:** `docker compose down -v && make train && docker compose up -d`. The `reconcile.py` guard detects this on agent wakeup and marks the investigation `aborted_stale`.

### Secrets not loading
**Cause:** `.env` file is missing or has typos. `Settings(extra="forbid")` crashes on unknown keys.
**Fix:** Compare `.env` against `.env.example`. Ensure `GOOGLE_API_KEY`, `POSTGRES_PASSWORD`, and `PROMOTION_API_KEY` are set.

---

## Health Endpoints

| Service | URL |
|---|---|
| service | `http://localhost:8000/health` |
| agent | `http://localhost:8001/health` |
| mlflow | `http://localhost:5000/health` |
| dashboard | `http://localhost:8501` |
