# Drift Triage Co-Pilot

Self-healing MLOps stack: a binary classifier trained on UCI Bank Marketing data, a FastAPI drift-detection platform, and a LangGraph supervisor agent that autonomously triages drift alerts, proposes remediation, and routes Production changes through human approval.

## Architecture

```
┌─────────────┐   predict   ┌─────────────────────────────────────────┐
│   Clients   │────────────▶│  service :8000                          │
└─────────────┘             │  POST /predict                          │
                            │  GET  /drift/report  (TTL cache 60s)    │
                            │  POST /promotion/promote  (gated)       │
                            └────────────┬────────────────────────────┘
                                severity │ change → webhook
                                         ▼
                            ┌─────────────────────────────────────────┐
                            │  agent :8001                            │
                            │  POST /webhook/drift  → open inv.       │
                            │  POST /hil/approve    → resume graph    │
                            │  GET  /investigations                   │
                            │                                         │
                            │  LangGraph supervisor:                  │
                            │  triage → action → (HIL?) → comms       │
                            │         → dispatch → END                │
                            └────────────┬────────────────────────────┘
                                         │ enqueue job
                                         ▼
                            ┌─────────────────────────────────────────┐
                            │  worker  (arq + Redis)                  │
                            │  replay_test | retrain | rollback       │
                            │  idempotency: SETNX per job key         │
                            │  DLQ: drift_actions:dlq                 │
                            └─────────────────────────────────────────┘
                                         │
                 ┌───────────────────────┼──────────────────┐
                 ▼                       ▼                   ▼
          Postgres 16           MLflow registry         Redis 7
          (predictions,         (model versions,        (queue,
           investigations,       artifacts,              dedup keys)
           checkpoints,          staging/prod)
           HIL approvals)

                            ┌─────────────────────────────────────────┐
                            │  dashboard :8501  (Streamlit)           │
                            │  Drift report · Investigations          │
                            │  HIL inbox · Queue status · Registry    │
                            └─────────────────────────────────────────┘
```

Secrets flow: `.env` (gitignored) → `Settings` class → injected via `Depends()`.

## Prerequisites

- Docker 24+ and Docker Compose v2
- `uv` ([install](https://docs.astral.sh/uv/))
- 8 GB RAM (MLflow + Postgres + models)

## Setup

```bash
git clone <repo> && cd drift-triage-copilot

# 1. Copy env template
cp .env.example .env

# 2. Start infrastructure
docker compose up -d postgres redis mlflow

# 3. Install backend deps and download data
make install
make data

# 4. Train and register model
make train
# Registers model in MLflow as Staging; promotes to Production if it passes the gate.

# 5. Start all services
docker compose up -d

# 6. Open dashboard
open http://localhost:8501
```

## Environment Variables

All secrets come from `.env` (gitignored). Copy `.env.example` and fill in the required values.

| Variable | Description | Required |
|---|---|---|
| `GOOGLE_API_KEY` | Gemini API key (primary LLM) | Yes |
| `POSTGRES_PASSWORD` | Postgres password | Yes |
| `PROMOTION_API_KEY` | Internal key for `/promotion/promote` endpoint (min 16 chars) | Yes |
| `GEMINI_MODEL` | Gemini model name (default: `gemini-2.5-flash`) | No |
| `OLLAMA_MODEL` | Ollama fallback model (default: `llama3`) | No |

For the full list of optional config variables, see `.env.example`.

## ML Narrative

### Feature engineering

- **`duration` dropped** — recorded after the call ends; leaks the target. Any model trained with it achieves near-perfect accuracy but collapses on new data.
- **`pdays==999` → `was_previously_contacted=0`** — 999 is a sentinel meaning "client was never contacted"; treating it as a continuous value distorts distance metrics.
- **`unknown` kept as a real category** — `unknown` is informative; clients who don't disclose job/marital status have different subscription rates than those who do.
- **Stratified 60/20/20 split, `random_state=42`** — preserves the ~11% positive-class imbalance in all splits.

### Model comparison

| Model | Val AUC | Val Recall | Val Precision | Val F1 |
|---|---|---|---|---|
| DummyClassifier (baseline) | 0.50 | — | — | — |
| LogisticRegression | _filled after train_ | — | — | — |
| RandomForestClassifier | _filled after train_ | — | — | — |
| GradientBoostingClassifier | _filled after train_ | — | — | — |

Best model registered in MLflow. Run `mlflow ui` (port 5000) to compare runs.

### Threshold tuning

Operating threshold = highest value where `recall >= 0.75` on the validation set. Bank marketing campaigns are recall-sensitive: missing a subscriber (false negative) wastes acquisition budget; the threshold rule captures this.

### Drift detection methodology

- **PSI** on each numeric feature: compares rolling window (last 500 predictions) distribution to training reference. PSI < 0.1 = stable, 0.1–0.25 = moderate, ≥ 0.25 = significant.
- **Chi-squared test** on each categorical feature: p < 0.05 = significant drift.
- **Output PSI**: PSI on predicted class proportions — catches silent drift where features look stable but model output has shifted.
- Overall severity = `max` across all individual feature results.

## Agent Narrative

### Supervisor topology

```
webhook → triage → should_act?
              ├─ yes → action → needs_approval?
              │         ├─ yes → pause_for_human (HIL) → resume → comms → dispatch → END
              │         └─ no  → comms → dispatch → END
              └─ no  → comms → END
```

### HIL flow

1. Action sub-agent proposes a Production-touching action (retrain/rollback).
2. Agent pauses; HIL approval request appears in dashboard.
3. Human approves or rejects in the HIL inbox.
4. **Staleness guard** checks `investigation.drift_report_id == latest_report.report_id` before executing — aborts if a newer drift event has arrived.
5. On approval: arq worker executes the job.

### Queue

- Library: `arq` (async, Redis-backed).
- Idempotency: `SETNX dispatch:{job_type}:{investigation_id}` with 24h TTL — duplicate dispatches are no-ops.
- Retries: 3 attempts, exponential backoff 1s → 2s → 4s.
- DLQ: `drift_actions:dlq` — visible in dashboard Queue tab.

### Checkpoint persistence

Agent state persists to Postgres via `langgraph-checkpoint-postgres`. Kill the agent container mid-investigation and restart it — investigation resumes from the last checkpoint, not from scratch.

## Demo Script

See [CLAUDE.md §21](CLAUDE.md) for the Friday 5-minute demo walkthrough.

## Deployment Notes

- `docker compose up` from clean clone after filling `.env`.
- Gate: all healthchecks pass before any service starts (see `depends_on` in `docker-compose.yaml`).
- See [RUNBOOK.md](RUNBOOK.md) for common failures and recovery.
