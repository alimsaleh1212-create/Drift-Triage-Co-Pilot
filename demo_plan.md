# Demo Workflow Walkthrough

The stack is up and a model is trained (Staging v2, logistic regression, threshold 0.385).
Walk through the full drift → webhook → agent pipeline step by step, inspecting logs at each stage.

---

## Step 1 — Verify the baseline (all services healthy)

```bash
docker compose ps
curl -s http://localhost:8000/health | python3 -m json.tool
curl -s http://localhost:8001/health | python3 -m json.tool
```

Expected: both return `{"status": "ok"}`.

---

## Step 2 — Open log streams in separate terminals

**Terminal A** — service logs (predictions + drift + webhook):
```bash
docker logs -f dt_service 2>&1 | grep -v "^INFO"
```

**Terminal B** — agent logs (investigation lifecycle):
```bash
docker logs -f dt_agent 2>&1
```

**Terminal C** — your main terminal (run commands here).

---

## Step 3 — Send a single test prediction (smoke test)

```bash
curl -s -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35, "job": "admin.", "marital": "married",
    "education": "university.degree", "default": "no",
    "housing": "yes", "loan": "no", "contact": "cellular",
    "month": "may", "day_of_week": "mon", "campaign": 1,
    "pdays": 999, "previous": 0, "poutcome": "nonexistent",
    "emp.var.rate": -1.8, "cons.price.idx": 92.893,
    "cons.conf.idx": -46.2, "euribor3m": 1.299, "nr.employed": 5099.1
  }' | python3 -m json.tool
```

Watch Terminal A for: `prediction.complete` event with `label` and `probability`.

---

## Step 4 — Check the baseline drift report

```bash
curl -s http://localhost:8000/api/v1/drift/report | python3 -m json.tool
```

Note the `severity` and `window_size`. Window < ~100 rows means PSI bins are unreliable — that's fine for the baseline check.

---

## Step 5 — Generate demo batch CSVs

```bash
python3 scripts/generate_demo_batches.py
ls -lh scripts/batches/
```

Produces 4 CSVs:

| File | Scenario | Expected agent action |
|------|----------|-----------------------|
| `normal_batch.csv` | Real rows, no shift | `monitor` / no webhook |
| `replay_drift_batch.csv` | Mild macro + categorical shift | `replay_test` |
| `retrain_drift_batch.csv` | Strong economic-regime shift | `retrain` |
| `rollback_drift_batch.csv` | Severe population shift | `rollback` |

---

## Step 6 — Inject normal batch (establish reference window)

```bash
python3 scripts/inject_demo_batch.py --batch scripts/batches/normal_batch.csv
```

Watch Terminal A: stream of `prediction.complete` events.
Then re-check drift — severity should stay `low`:

```bash
curl -s http://localhost:8000/api/v1/drift/report \
  | python3 -c "import sys,json; r=json.load(sys.stdin); print('severity:', r['severity'], '| window:', r['window_size'])"
```

---

## Step 7 — Inject drift batch and watch the webhook fire

```bash
python3 scripts/inject_demo_batch.py --batch scripts/batches/retrain_drift_batch.csv
```

**Terminal A** — watch for:
- `drift.webhook.sent` — severity changed, agent notified (note `severity`, `previous`, `report_id`)

**Terminal B** — watch for immediately after:
- `webhook.received` — agent got the payload
- `supervisor.route` — graph routing decisions (triage → action → …)
- `llm.call` — which LLM provider + schema
- Tool calls: `inspect_drift`, `propose_action`, `compose_summary`

---

## Step 8 — Poll drift report to confirm severity changed

```bash
for i in 1 2 3; do
  curl -s http://localhost:8000/api/v1/drift/report \
    | python3 -c "import sys,json; r=json.load(sys.stdin); print('severity:', r['severity'], '| window:', r['window_size'], '| top PSI:', max((x['psi'] for x in r['psi_results']), default=0))"
  sleep 5
done
```

---

## Step 9 — Inspect the investigation in Postgres

```bash
docker exec postgres psql -U postgres -d drift_triage -c \
  "SELECT id, status, created_at FROM investigations ORDER BY created_at DESC LIMIT 5;"
```

If retrain was proposed (HIL approval needed):
```bash
docker exec postgres psql -U postgres -d drift_triage -c \
  "SELECT id, action, status, model_version FROM hil_approvals ORDER BY created_at DESC LIMIT 3;"
```

---

## Step 10 — Read the full agent trajectory for one investigation

```bash
INV_ID="<paste id from Step 9>"
docker logs dt_agent 2>&1 | grep "$INV_ID"
```

Shows the complete path: triage → action → HIL pause (if production-touching).

---

## Step 11 — (Optional) Try replay_drift for a lighter action

```bash
python3 scripts/inject_demo_batch.py --batch scripts/batches/replay_drift_batch.csv
```

Agent should propose `replay_test` — no HIL, dispatches directly to arq worker:

```bash
docker logs dt_worker 2>&1 | tail -30
```

---

## Key log events cheat-sheet

| Event | Container | Meaning |
|-------|-----------|---------|
| `prediction.complete` | dt_service | Inference done |
| `drift.webhook.sent` | dt_service | Severity changed, agent notified |
| `drift.webhook.failed` | dt_service | Agent unreachable (retry queued) |
| `webhook.received` | dt_agent | Agent received the drift alert |
| `supervisor.route` | dt_agent | Graph routing decision |
| `llm.call` | dt_agent | LLM provider + schema used |
| `tool.validation_error` | dt_agent | Bad tool args from LLM |
| `graph.run_failed` | dt_agent | Investigation crashed |

---

## Quick reset (fresh demo from clean state)

```bash
docker exec postgres psql -U postgres -d drift_triage \
  -c "DELETE FROM predictions;" \
  -c "DELETE FROM investigations;" \
  -c "DELETE FROM hil_approvals;" \
  -c "DELETE FROM drift_alert_state;"
```
