# Architecture — Drift Triage Co-Pilot

## Sequence: Prediction → Drift Detection → Webhook

```
Client          service         Postgres          agent
  |                |                |               |
  |--POST /predict→|                |               |
  |                |--INSERT prediction→|           |
  |                |←200 PredictResponse            |
  |                |                |               |
  |--GET /drift/report→|            |               |
  |                |--SELECT last 500→|             |
  |                |  compute PSI/chi2/output-PSI   |
  |                |←DriftReport    |               |
  |                |                |               |
  |                |--SELECT drift_alert_state→|     |
  |    [severity changed]           |               |
  |                |--POST /webhook/drift (BG)-----→|
  |                |                |               |--open investigation
  |                |--UPSERT drift_alert_state←|     |
```

## Sequence: Agent Triage → HIL → Dispatch

```
agent (graph)         Postgres          agent API       Dashboard       worker
     |                    |                 |               |               |
     |--inspect_drift--→service             |               |               |
     |←DriftReport         |                |               |               |
     |--triage (LLM)       |                |               |               |
     |--propose_action      |                |               |               |
     |--request_hil_approval→|              |               |               |
     |                    |--INSERT hil_approval             |               |
     |--update_dashboard--→|                |               |               |
     |                    |                 |←GET /invest.--→               |
     |                    |                 |               |--shows HIL box |
     |                    |                 |←POST /hil/approve              |
     |  [resume graph]     |                |               |               |
     |--[staleness check]  |                |               |               |
     |--dispatch_action    |                |               |               |
     |                    |                 |               |←--enqueue job--→
     |--comms (LLM)        |                |               |               |
     |--update_dashboard--→|                |               |               |
                                                                    |--execute job--|
                                                                    |  (retrain /   |
                                                                    |   rollback)   |
```

## Data Flow: Secrets

```
.env (GOOGLE_API_KEY, POSTGRES_PASSWORD, PROMOTION_API_KEY)
  └─▶ Settings class (lru_cache) → injected via Depends()
        └─▶ Services read secrets at startup; .env is gitignored
```

## Component Responsibilities

| Component | Owns | Exposes |
|---|---|---|
| `service` | Classifier loading, rolling-window predictions, PSI/chi² computation, drift cache, promotion gate | `POST /predict`, `GET /drift/report`, `POST /promotion/promote` |
| `agent` | LangGraph graph, idempotent drift webhook intake, investigation lifecycle, HIL approval routing | `POST /webhook/drift`, `POST /hil/approve`, `GET /investigations` |
| `worker` | Slow job execution (retrain/rollback/replay), idempotency SETNX, DLQ | arq queue consumer |
| `dashboard` | Read-only API consumer, HIL approval UI | Streamlit :8501 |
| `mlflow` | Model registry, run tracking, artifact storage | HTTP :5000 |
| `postgres` | Predictions, investigations, HIL approvals, LangGraph checkpoints, MLflow metadata | TCP :5432 |
| `redis` | arq job queue, dedup SETNX keys, DLQ list | TCP :6379 |
