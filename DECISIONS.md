# Architecture Decisions

| Decision | Choice | Rationale |
|---|---|---|
| LLM primary | Gemini 2.5 Flash / Pro | Quality + structured output; affordable flash for high-frequency triage calls |
| LLM fallback | Ollama (llama3) | Resilience when Gemini is down; no external dependency at runtime |
| Webhook vs poll | Push (platform → agent) | Lower latency; simpler agent state; avoids polling interval tuning |
| Queue library | arq (async, Redis) | Lightweight, async-native, built-in retry/DLQ, no Celery broker overhead |
| Agent checkpoint | Postgres (langgraph-checkpoint-postgres) | Brief requirement; survives restarts; queryable; same infra as rest of stack |
| Model registry | MLflow with Postgres backend | Full versioning, staging/production transitions, run comparison UI |
| Secrets | HashiCorp Vault (AppRole) | No secrets in env or git; least-privilege per service; dev server in Docker |
| Promotion gate | HTTP endpoint + API key | Programmatic gate enforces checklist; only worker knows the key (from Vault) |
| Promotion — can it bypass agent? | Technically yes; practically no | Requires API key only the worker has + recorded HIL approval row in DB. Documented here as explicit decision. |
| Streamlit refresh | st_autorefresh every 5s | Demo requires live visibility of drift/HIL changes without websockets |
| arq vs Celery | arq | Celery requires a broker process and serialization config; arq is async-native and ~200 LOC to configure |
| PSI bins | 10 quantile bins | Standard for PSI; quantile bins avoid empty-bin issues on skewed distributions |
| Chi2 alpha | 0.05 (warn), 0.01 (high) | p < 0.05 is the standard significance threshold; 0.01 reserves "high" for strong evidence |
| Drift window | 500 predictions | Large enough for stable distribution estimates; small enough to catch recent changes |
| ML split | 60/20/20 | Val set used for threshold tuning; test set held out for reference stats and fidelity replay |
| Streamlit env vars | `os.getenv()` directly | Streamlit runs as a separate process without pydantic-settings; these are service URLs (not secrets), so the risk is low. Documented as an explicit exception to the "all config through Settings" rule. |
