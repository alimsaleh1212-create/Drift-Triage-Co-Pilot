# CLAUDE.md — Drift Triage Co-Pilot

Combined behavioural + engineering standards for this project. Synthesised from
the AIE Bootcamp Week 5 brief, the Engineering Standards Companion Guide, the
AIE Coding Guidelines, and Week 4 Code Review Lessons Learned. Follow
automatically on every file you create or modify.

> **Engineering is the discipline of writing code that other people can change
> without fear.** — Hasan, Companion Guide

---

## Part I — How to Think

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- **Defend every line.** If you cannot explain in one sentence why a file,
  function, import, or dependency exists, remove it or understand it first.
  Never commit black-box AI-generated code you cannot read line by line.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- Remove imports/variables/functions that **your** changes made unused — nothing more.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

For multi-step tasks, state a brief plan up front:

```
1. [step] → verify: [check]
2. [step] → verify: [check]
```

Each stage in `plan.md` ends with an explicit validation gate. Do not advance
until the gate is green.

---

## Part II — Architecture Patterns (the Companion Guide)

These nine patterns are not optional. They reinforce each other: async needs DI
to manage shared resources cleanly; DI needs lifespan singletons to know what
to inject; singletons need a typed Settings class to decide what to load;
errors need types so a tool failure has a shape; tests need DI to mock
dependencies.

### 5. Async All the Way Down

**Every route, tool, and external call is `async`. No blocking I/O in the
request path.**

The agent loop and model service are almost entirely I/O — they wait for the
LLM, Postgres, Redis, MLflow, and webhooks. Python has one event loop per
process; one blocking call freezes every other in-flight request.

```python
# ✗ Looks fine. It is not.
@app.post("/predict")
async def predict(payload: PredictRequest):
    result = requests.post(MODEL_URL, json=payload.dict())  # blocks
    return result.json()

# ✓ Real async. Real concurrency.
@app.post("/predict")
async def predict(payload: PredictRequest):
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(MODEL_URL, json=payload.model_dump())
    return r.json()
```

**Pitfalls:**

- `time.sleep` in async code → use `await asyncio.sleep`.
- `requests` anywhere in a request path → replace with `httpx`.
- CPU-bound work (sklearn `model.predict`) inside the event loop
  → wrap in `await asyncio.to_thread(...)`. The bank classifier is light, but
  wrap it anyway for consistency and to future-proof.
- Always pass a `timeout=` to `httpx.AsyncClient` — a hung connection hangs
  the whole request indefinitely.

### 6. Dependency Injection — No Globals

**Declare what you need. Let FastAPI hand it to you.**

Globals are untestable, leak across requests, and hide initialisation order.
FastAPI's `Depends()` is the cleanest DI in any Python web framework.

```python
# deps/db.py
async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session

# deps/classifier.py — singleton, built in lifespan
def get_classifier(request: Request) -> Pipeline:
    return request.app.state.classifier

# deps/agent.py — singleton, built in lifespan
def get_agent(request: Request) -> CompiledGraph:
    return request.app.state.agent

# routers/prediction.py
@router.post("/predict")
async def predict(
    payload: PredictRequest,
    classifier: Pipeline = Depends(get_classifier),
    session: AsyncSession = Depends(get_session),
) -> PredictResponse:
    ...
```

Sessions are scoped to the request via the `yield` — they open and close
automatically. In tests:
`app.dependency_overrides[get_classifier] = lambda: dummy_pipeline` — no
monkey-patching, no source edits.

### 7. Singletons via Lifespan

Some objects exist exactly once per process. Build them on startup, attach
them to `app.state`, dispose on shutdown, expose them via dependencies.

```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.engine = create_async_engine(settings.async_database_url)
    app.state.classifier = load_classifier(settings.model_path)
    app.state.redis = arq.create_redis_pool(settings.redis_url)
    app.state.drift_detector = DriftDetector(settings)
    app.state.llm_primary = build_gemini_client(settings)
    app.state.llm_fallback = build_ollama_client(settings)
    yield
    await app.state.engine.dispose()
    await app.state.redis.close()
```

**Rule of thumb:**

| Lifetime           | Mechanism                          | Examples                                                |
| ------------------ | ---------------------------------- | ------------------------------------------------------- |
| Per process        | lifespan + `app.state`             | DB engine, classifier, Redis pool, LLM clients, drift detector |
| Per request        | `yield` in a `Depends()`           | DB session, transaction                                 |
| Per call           | nothing                            | values computed from inputs                              |

### 8. Caching — `lru_cache` and TTL Caches Where They Pay Off

The wrong cache is worse than no cache. Each cache has an explicit invalidation
policy.

| Subject                       | Mechanism                                       | Why                                          | Invalidation                       |
| ----------------------------- | ----------------------------------------------- | -------------------------------------------- | ---------------------------------- |
| `Settings`                    | `@lru_cache(maxsize=1)`                         | Read env once at startup                     | Process restart                    |
| Gemini / Ollama LLM clients  | `@lru_cache(maxsize=1)`                         | Auth + HTTP session expensive                | Process restart                    |
| sklearn classifier (joblib)   | Loaded in lifespan → `app.state.classifier`     | Loading per request is the named antipattern | Redeploy                           |
| Drift reference statistics    | Loaded once at startup → `app.state.ref_stats`   | Re-computing from training data is expensive | Retrain                            |
| Model registry metadata       | `TTLCache(maxsize=64, ttl=60)`                  | Versions don't change often                  | TTL; key = model name              |

**Implementation rule — thundering herd:** TTL caches in async code need an
`asyncio.Lock` with a double-check inside the lock.

```python
drift_cache = TTLCache(maxsize=64, ttl=60)
drift_lock = asyncio.Lock()

async def get_latest_drift_report(model_name: str) -> DriftReport:
    key = model_name
    if key in drift_cache:
        return drift_cache[key]
    async with drift_lock:
        if key in drift_cache:
            return drift_cache[key]
        result = await fetch_drift_report(model_name)
        drift_cache[key] = result
        return result
```

Caches live in **one** module per subject (the module that owns the resource),
never scattered. Document every TTL choice in the README.

**Do not** `lru_cache` anything taking mutable args, anything that should
expire, or any function whose cache key is not its inputs.

### 9. Configuration — `pydantic-settings`, Not Magic Strings

One `Settings` class. Every value typed. Missing required values fail at
startup. The rest of the codebase imports from `Settings` — never from
`os.environ` directly.

Secrets come from `.env` (gitignored) in development. For production,
migrate to HashiCorp Vault — see `DECISIONS.md`.

```python
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
    )

    # Secrets (from .env — gitignored)
    google_api_key: str = Field(..., min_length=1)
    postgres_password: str = Field(..., min_length=1)
    promotion_api_key: str = Field(..., min_length=16)

    # Non-secret config — safe to come from env vars or defaults
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_user: str = "drift_triage"
    postgres_db: str = "drift_triage"

    redis_url: str = "redis://redis:6379"
    mlflow_tracking_uri: str = "http://mlflow:5000"

    gemini_model: str = "gemini-2.5-flash"
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "llama3"

    random_state: int = 42
    test_size: float = 0.2
    val_size: float = 0.2
    min_recall: float = 0.75
    drift_psi_warn: float = 0.1
    drift_psi_high: float = 0.25
    drift_chi2_alpha: float = 0.05
    drift_window_size: int = 500

    redis_queue_name: str = "drift_actions"
    redis_max_retries: int = 3
    redis_retry_delay_base: float = 1.0

    service_url: str = "http://service:8000"
    agent_url: str = "http://agent:8001"

    @property
    def async_database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
```

**`extra="forbid"` is mandatory.** A typo in `.env` must crash at startup,
not silently leave a `None` for two weeks.

**`.env.example` contains every variable with placeholder values.** No real
secrets ever appear in `.env.example` or any file committed to git. `.env`
is in `.gitignore`.

Tests construct `Settings(google_api_key="test", ...)` directly — no env
variables required.

**Production note:** For production deployment, migrate secrets to
HashiCorp Vault. See `DECISIONS.md` for the migration path.

### 10. Type Hints, Pydantic, and the Boundary

**Validate at the edges. Trust your types inside.**

| Boundary                                | Pydantic model                                     |
| --------------------------------------- | -------------------------------------------------- |
| Prediction request/response             | `PredictRequest`, `PredictResponse`                |
| Drift report / alert                    | `DriftReport`, `DriftAlert`, `PSIResult`, `Chi2Result` |
| Promotion request/response              | `PromotionRequest`, `PromotionResponse`            |
| Webhook payload (platform → agent)     | `DriftWebhookPayload` (versioned in `contracts/v1/`) |
| Agent tool input/output                 | `InspectDriftInput`, `ReplayTestInput`, etc.        |
| Agent state                             | `AgentState` (TypedDict for LangGraph)              |
| HIL approval                            | `HILApprovalRequest`, `HILApprovalResponse`        |
| Job result (queue worker)              | `JobResult`, `JobError`                              |

```python
class DriftWebhookPayload(BaseModel):
    """Versioned contract: platform → agent on drift severity change."""
    version: Literal["v1"] = "v1"
    drift_report_id: str = Field(..., min_length=1)
    model_name: str = Field(..., min_length=1)
    model_version: int = Field(..., ge=1)
    severity: Literal["low", "medium", "high"]
    psi_results: list[PSIResult]
    chi2_results: list[Chi2Result]
    output_drift: float
    timestamp: datetime
```

Once a value crosses the boundary as a validated model, **no inner function
re-checks `isinstance(x, dict)` or `if not x: return None`**. The 80%-defence,
20%-logic function is the antipattern.

Type hints are required on every function signature. `mypy --strict` runs in
pre-commit and CI.

### 11. Errors, Retries, and Failure Isolation

Three layers — you need all three.

**Layer 1 — Timeouts.** Every external call gets a timeout. No exceptions.

```python
async with httpx.AsyncClient(timeout=10.0) as client:
    r = await client.get(url)
```

**Layer 2 — Retries with backoff (transient only).** Use `tenacity`. Retry on
network / timeout / 5xx; **never** on 4xx (it'll fail the same way forever).

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
    reraise=True,
)
async def fetch_drift_report(model_name: str) -> DriftReport:
    ...
```

**Layer 3 — Failure isolation in the agent loop.** A tool failure must **not**
crash the agent. Return a structured `ToolError` so the LLM can reason about
the failure.

```python
class ToolError(BaseModel):
    error: str
    retryable: bool

async def inspect_drift(args: InspectDriftInput) -> InspectDriftResult | ToolError:
    try:
        return await fetch_drift(args)
    except httpx.HTTPStatusError as e:
        return ToolError(error=f"upstream {e.response.status_code}", retryable=False)
    except (httpx.TimeoutException, httpx.NetworkError) as e:
        return ToolError(error=f"unreachable: {e}", retryable=True)
```

This is what `safe_run()` on `BaseTool` enforces for every tool — exceptions
become `ToolResult(ok=False, error=...)`, never propagate into the LangGraph
loop.

**What never to do:**

- Bare `except:` or `except Exception: pass`. Catch specific types.
- Retry without a max-attempt cap (locks up your event loop).
- Let the webhook failing break the prediction response. Webhook failure is
  logged; the API still returns the prediction.
- Leak stack traces / file paths to clients. Trace goes to logs only.

### 12. Code Hygiene

**Project layout — one concern per file.** A 600-line `main.py` is a sign no
one is in charge of structure.

**File names describe content.** Banned: `utils.py`, `helpers.py`, `misc.py`,
`stage1.py`, `thing.py`. If you can't pick a descriptive name, the file is
doing too many unrelated things — split it.

**Every endpoint lives in an `APIRouter`** — never in `main.py`. Adopt this on
day one; the cost is zero, the benefit is that endpoint #20 doesn't require
a restructure.

**Logging — never `print()`.** Use a structured logger; every log line is a
JSON object with named fields.

```python
import structlog

log = structlog.get_logger()

async def run_investigation(alert: DriftWebhookPayload) -> None:
    log.info("investigation.start", alert_id=alert.drift_report_id, severity=alert.severity)
    try:
        result = await agent.ainvoke({"alert": alert})
        log.info("investigation.complete", alert_id=alert.drift_report_id)
    except Exception:
        log.exception("investigation.failure", alert_id=alert.drift_report_id)
        raise
```

Levels: `DEBUG` (internals) · `INFO` (milestones) · `WARNING` (recoverable) ·
`ERROR` (failed op) · `CRITICAL` (service down). Never log passwords, JWTs,
PII, or API keys.

### 13. Tests — At Least the Critical Path

You will not get to 100% coverage and you should not try. You **must** have a
small set of tests that runs in seconds, fails loudly when something
important breaks, and runs automatically on every commit.

**Four categories for this project:**

1. **Pydantic schemas.** Cheap, high-value. Test both valid and invalid input.
   ```python
   def test_drift_webhook_rejects_invalid_severity():
       with pytest.raises(ValidationError):
           DriftWebhookPayload(severity="critical", ...)
   ```

2. **Tools — mock the LLM, mock the API, test the logic.** Each tool should
   be testable without calling the real Gemini or hitting the real model service.
   ```python
   async def test_inspect_drift_handles_api_failure(monkeypatch):
       async def boom(*a, **kw): raise httpx.TimeoutException("t/o")
       monkeypatch.setattr("app.agent.tools.inspect_drift._fetch_report", boom)
       result = await inspect_drift(args)
       assert isinstance(result, ToolError)
       assert result.retryable is True
   ```

3. **Agent snapshot trajectory tests.** Mock the LLM to return predetermined
   tool calls. Inject a drift webhook, assert agent follows the exact path
   through triage → action → comms. Snapshot the full state at each step.
   Any routing change = test failure = must update snapshot.
   **These run without an API key.**

4. **End-to-end — one happy path through the whole agent.** With external
   APIs mocked, run a full request and assert the right tools fire and the
   response is well-formed.

**Coverage targets:**

- Critical paths (promotion gate, drift detection, agent routing): ≥ 95%
- Tools, agent graph nodes: ≥ 90%
- Overall backend: ≥ 80%

Tests run in CI on every push. A test that doesn't run is a test that doesn't
exist.

**Snapshot trajectory fixtures live in `tests/agent/snapshots/`.** One JSON
file per recorded scenario. Format:

```json
{
  "scenario": "high_severity_retrain_path",
  "webhook": { "version": "v1", "severity": "high", "...": "..." },
  "llm_responses": [
    {"node": "triage", "tool_calls": [{"name": "inspect_drift", "args": {...}}]},
    {"node": "action", "tool_calls": [{"name": "propose_action", "args": {"action": "retrain"}}]},
    {"node": "comms",  "tool_calls": [{"name": "compose_summary", "args": {...}}]}
  ],
  "expected_state_after": [
    {"node": "triage",  "fields": {"severity": "high", "drift_features": ["euribor3m"]}},
    {"node": "action",  "fields": {"proposed_action": "retrain", "needs_approval": true}},
    {"node": "comms",   "fields": {"summary_present": true}}
  ]
}
```

A `FakeLLM` in `tests/agent/conftest.py` replays `llm_responses` in order. CI
runs these without an `API_KEY`. Any routing change → snapshot mismatch →
update the JSON and explain in the PR description.

---

## Part III — Domain-Specific Rules (Drift Triage Co-Pilot)

### 14. LLM Provider — Gemini Primary with Ollama Fallback

This project uses **Google Gemini** as the primary LLM, falling back to
**Ollama** (local) if Gemini is unavailable.

- Primary client: `google-generativeai` Python SDK (Gemini, configured via
  `GEMINI_MODEL` setting, default `gemini-2.5-flash`).
- Fallback client: `httpx` to local Ollama service (configured via
  `OLLAMA_MODEL` setting, default `llama3`).
- Auth: `GOOGLE_API_KEY` from `.env` via `Settings`.
- **Single model.** One Gemini model for all calls — triage, action, comms.
  No cheap/strong split. If a more capable model is needed, change
  `GEMINI_MODEL` in `.env`.
- The fallback is automatic: on any exception from Gemini (timeout, network
  error, 5xx), fall back to Ollama.
- Use `response_mime_type="application/json"` with `response_schema=PydanticModel`
  for structured extraction. **Never** parse free-form LLM text with regex or
  string splitting.
- **Separate prompt layers.** System prompt = role + format + invariants
  (static); user prompt = the varying data only.
- Cache the Gemini client with `@lru_cache(maxsize=1)`.
- Every call has `max_output_tokens`, `timeout`, and tenacity retries on
  transient errors.
- Log which provider served each call for observability.
- Log prompts and responses with sensitive fields scrubbed.

**Fallback pattern:**

```python
async def call_llm(prompt: str, schema: type[M]) -> M:
    try:
        result = await _call_gemini(prompt, schema)
        log.info("llm.call", provider="gemini", schema=schema.__name__)
        return result
    except Exception as exc:
        log.warning("llm.fallback", provider="gemini", error=str(exc))
        result = await _call_ollama(prompt, schema)
        log.info("llm.call", provider="ollama", schema=schema.__name__)
        return result
```

### 15. LLM Input Security — Prompt Injection Prevention

User input and drift data flow into LLM prompts. Every tool must enforce these
eight rules. The implementation lives in `src/drift_triage/agent/security.py`.

1. **Sanitize before format().** Call `_sanitize_query()` on every external
   string before interpolating it into any prompt template (normalises
   whitespace, strips control chars, truncates, removes `system:` / `assistant:`
   leak triggers).
2. **Delimit user content.** Wrap external text in `<external_data>...</external_data>`
   tags in every prompt template. Never embed raw external text adjacent to
   instructions.
3. **Sanitize LLM string outputs before re-use.** Any string from a
   model output that flows into another prompt or a tool argument runs through
   `_sanitize_feature_string()`.
4. **`max_output_tokens` on every LLM call.** Caps exfiltration attempts.
5. **No `eval()` / `exec()` on LLM output.** Ever. Tool args are JSON-parsed
   and Pydantic-validated; failures return a structured error to the LLM for
   retry.
6. **Log suspicious patterns.** Patterns (`ignore previous`, `you are now`,
   `system prompt`, `\n\nHuman:`) are logged at WARNING but not rejected.
7. **Pydantic is the fence.** Every tool input is a Pydantic model;
   structurally invalid args never reach the implementation.
8. **Tool allowlist.** The agent rejects any tool name outside the registered
   set, even if the LLM hallucinates one.

### 16. Agent — LangGraph Supervisor + 3 Sub-Agents

The triage copilot uses a **true supervisor topology** — three sub-agents
(triage, action, comms), not a chain. Each sub-agent has a distinct role:

- **Triage sub-agent**: Receives the drift alert, analyses the drift report
  (PSI values, chi² p-values, output distribution shift), determines severity,
  identifies which features drifted and by how much, hypothesises root cause.
- **Action sub-agent**: Proposes a response action (replay test set, retrain,
  rollback to previous version, or no action needed). If the action touches
  Production, pauses for human approval (HIL).
- **Comms sub-agent**: Formats the investigation summary for the dashboard,
  writes the HIL approval request, and composes the final resolution message.

**LangGraph topology** (`src/drift_triage/agent/graph.py`):

```
webhook_received → triage → should_act?
                              ├─ yes → action → needs_approval?
                              │                 ├─ yes → pause_for_human → (resume) → comms → dispatch → END
                              │                 └─ no  → comms → dispatch → END
                              └─ no  → comms → END
```

Every tool subclasses `BaseTool` (`src/drift_triage/agent/tools/base.py`):

```python
class BaseTool(ABC, Generic[InputT, OutputT]):
    name: ClassVar[str]
    input_schema: ClassVar[type[BaseModel]]
    output_schema: ClassVar[type[BaseModel]]

    @abstractmethod
    async def run(self, args: InputT) -> OutputT: ...

    async def safe_run(self, raw_args: dict) -> ToolResult[OutputT]:
        """Validate raw_args → run() → wrap exceptions. Never raises."""
        ...
```

`safe_run()` is the **only** entry point the LangGraph node calls. It
validates with `input_schema`, calls `run()`, and wraps exceptions as
`ToolResult(ok=False, error=...)`. Validation failures return a structured
error to the LLM for retry — never as an exception to the user.

**Agent state persists across restarts.** Postgres via
`langgraph-checkpoint-postgres`. Killing the agent mid-investigation and
restarting it must resume from the last checkpoint, not start over.

**Prompts as code.** Every prompt is a `.md` file in
`src/drift_triage/agent/prompts/`. Never inline strings. System prompt = role +
format + invariants (static); user prompt = the varying data only.

**Brief's "think-about" questions — answered, with implementation pointers:**

| Question                                                                                       | Answer                                                                                                                                                                            | File                                                              |
| ---------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| How do you keep checkpoint store and registry in sync when one rolls back and the other doesn't? | On every promote/rollback, agent emits a `RegistryEvent` to its own log; on wakeup, agent reconciles its current investigation's `model_version` against MLflow before any tool runs. | `agent/reconcile.py`                                              |
| Wakeup and the model URI it was investigating no longer exists?                                 | Reconcile step detects missing version, marks investigation `aborted_stale`, posts a comms summary, exits cleanly. No retry on a missing artifact.                                | `agent/reconcile.py`, `agent/graph.py` (entry guard)              |
| Two retries of the same retrain — guarantee one training?                                       | Idempotency key `retrain:{investigation_id}`; worker `SETNX` on `dispatch:{key}` with TTL = max-job-runtime + buffer; second dispatch is a no-op.                                  | `worker/dedup.py`, `worker/main.py`                                |
| Stale HIL approval (newer drift event arrived)?                                                 | Before executing approved action, agent checks `investigation.drift_report_id == latest_drift_report_id`. If false, action aborts and dashboard surfaces a warning.                | `agent/staleness.py`, `agent/graph.py` (post-approval guard)       |
| Can the promotion endpoint be called without going through the agent — and should it?           | Technically yes (it's HTTP); practically no — it requires `X-Promotion-Key` which only the worker reads from Settings (.env), and the gate requires a recorded HIL approval row. **Should not.** Documented in `DECISIONS.md`. | `service/routers/promotion.py`, `DECISIONS.md`                     |

If you change one of these answers, update the table.

### 17. ML Pipeline — UCI Bank Marketing

**Dataset:** `bank-additional-full.csv` — ~41,188 rows × 20 features.
Target: did the client subscribe to a term deposit? (~11% positive).

**Mandatory preprocessing:**

- **Drop `duration`.** It's recorded after the call ends and leaks the target.
  Removing it is non-negotiable.
- **Treat `pdays==999` as a sentinel.** Create a binary flag
  `was_previously_contacted` (1 if pdays < 999, 0 if pdays == 999). Optionally
  bin or log-transform the remaining pdays values.
- **Treat `'unknown'` as a real category**, not as missing data — it is
  informative. OneHotEncoder with `handle_unknown='infrequent_if_exist'`.
- **Stratified 60/20/20 split**, `random_state=42`.

**Feature pipeline** is an sklearn `Pipeline` with `ColumnTransformer` doing
imputation + scaling + one-hot encoding **inside** the pipeline. Never
pre-transform the data — that's how leakage gets in.

**Threshold tuning:** Find the highest threshold where recall >= 0.75 on the
validation set. This operating threshold is stored with the model in MLflow
and used at serving time.

**Model registration in MLflow:**

- Binary (joblib pipeline)
- Schema (input feature names and types)
- Model card: version hash, environment fingerprint, training date, metrics,
  operating threshold, dataset hash

**Reference statistics for drift:** At training time, compute and store:
- Mean and standard deviation of each numeric feature (for PSI)
- Category frequency distribution of each categorical feature (for chi²)
- Predicted class proportions on the test set (for output drift)

These reference stats are loaded at serving time and compared against the
rolling window of recent predictions.

**Drift detection:**

- **PSI (Population Stability Index)** on each numeric feature. Compare the
  current rolling window distribution (last N predictions) to the reference
  distribution from training.
  - PSI < 0.1: no significant change
  - 0.1 ≤ PSI < 0.25: moderate change
  - PSI ≥ 0.25: significant change → severity upgrade
- **Chi-squared test** on each categorical feature. Compare current category
  frequencies to reference. p < 0.05 → significant change.
- **Output distribution drift.** Compare current predicted-class proportions
  to reference proportions. Use PSI on the output distribution.

**Severity determination:**

| Metric               | Low       | Medium          | High            |
| -------------------- | --------- | --------------- | --------------- |
| PSI (numeric)        | < 0.1     | 0.1 – 0.25      | ≥ 0.25          |
| Chi² p-value         | > 0.05    | 0.01 – 0.05     | < 0.01          |
| Output PSI           | < 0.1     | 0.1 – 0.25      | ≥ 0.25          |

Overall severity = max across all features.

**Webhook emission:** When drift report severity changes (e.g., low → medium),
POST a `DriftWebhookPayload` to the agent's `/webhook/drift` endpoint.
Use `httpx.AsyncClient` with timeouts and tenacity retries. Delivery runs in
`BackgroundTasks` — webhook failure is logged, never breaks the prediction.

### 18. Redis Queue — arq with Idempotency and DLQ

Slow tools (replay test set, retrain, rollback) are dispatched through a
Redis-backed queue, not executed inline.

**Queue library:** `arq` — async, Redis-backed, lightweight, built-in
retry/DLQ support.

**Job structure:**

```python
class DriftJob(BaseModel):
    job_type: Literal["replay_test", "retrain", "rollback"]
    investigation_id: str
    idempotency_key: str  # e.g., "retrain:inv_abc123"
    payload: dict
    max_retries: int = 3
```

**Idempotency keys.** Every job carries a key like `retrain:{investigation_id}`.
Duplicate dispatches (e.g., two retries of the same retrain) are deduplicated.
Before executing, the worker checks if a job with the same idempotency key has
already completed or is in progress.

**Exponential-backoff retries.** 3 attempts with delays: 1s → 2s → 4s.

**Dead-letter queue (DLQ).** After max retries, the job moves to
`drift_actions:dlq`. The dashboard surfaces DLQ contents.

**Three job types:**

| Job type      | What it does                                                     |
| ------------- | ---------------------------------------------------------------- |
| `replay_test` | Run the current model on the held-out test set, compare metrics  |
| `retrain`     | Full training pipeline on current data, register as Staging     |
| `rollback`    | Re-promote previous stable Production version                    |

### 19. Promotion Gate

The platform must **refuse** to promote any version to Production except
through a programmatic gate that asserts the day-4 promotion checklist.

**`POST /promotion/promote`** — internal-only endpoint (requires
`promotion_api_key` in header).

The gate asserts:

1. Target model version exists and is in Staging.
2. AUC >= current Production model's AUC (or within configurable tolerance).
3. Recall >= 0.75 at the operating threshold.
4. No higher-severity drift event has occurred since the investigation was
   opened (staleness guard).
5. Human approval has been recorded in the HIL system.

On success: move model to Production in MLflow, record promotion in Postgres,
log the event.

**The promotion endpoint cannot be called without going through the agent.**
It requires an internal API key that only the agent/worker possesses. This is
documented in `DECISIONS.md`.

### 20. Human-in-the-Loop (HIL)

Before any action that touches Production (retrain, rollback, promote):

1. **Action sub-agent pauses.** The LangGraph state suspends with
   `awaiting_approval: True`.
2. **HIL approval request appears in the dashboard.** Shows:
   investigation ID, drift severity, proposed action, model version affected,
   who requested it, timestamp.
3. **Human approves or rejects via the dashboard.** Dashboard calls the
   agent's HIL approval endpoint.
4. **Staleness guard.** Before executing the approved action, the agent
   verifies that `investigation.drift_report_id == latest_drift_report.id`. If
   a newer drift event has arrived, the action is aborted and the dashboard
   shows a warning.

### 21. Drift Narrative & Live Demo

For the Friday presentation, the macroeconomic features (`euribor3m`,
`cons.price.idx`) are real economic indicators.

**Demo script:**

1. Start with normal predictions on the test split.
2. Shift one numeric feature (e.g., inject `euribor3m` values +2 std devs).
3. Shift one categorical feature (e.g., change `job` distribution).
4. Watch: platform detects drift → webhook → agent opens investigation →
   triage → action proposes response → HIL approval → queue dispatches →
   dashboard reflects the outcome.
5. Show one CI failure on a snapshot trajectory regression.
6. Show one specific real bug you hit and fixed.

### 22. Streamlit Dashboard

The dashboard surfaces both halves:

- **Registry view:** Current Production model version, Staging model(s),
  metric comparison, promotion history.
- **Drift monitoring:** Real-time drift report — PSI bars, chi² p-values,
  output distribution shift, severity badges.
- **Agent investigations:** Open investigations, resolved investigations,
  investigation detail drill-down with full trajectory.
- **Queue status:** arq queue depth, active jobs, DLQ contents, job history.
- **HIL inbox:** Pending approval requests with Approve/Reject buttons,
  investigation context, proposed action details.

**Auto-refresh every 5 seconds** — the dashboard polls the model service and
agent APIs for live updates.

### 23. Contracts Between Platform and Agent

The platform and agent agree on a contract:

- **Platform → Agent:** HTTP POST webhook to `/webhook/drift` on every
  severity change. Payload is `DriftWebhookPayload` (versioned in
  `contracts/v1/drift_webhook.json`).
- **Agent → Platform:** HTTP POST to `/promotion/promote` when an action is
  approved. Payload is `PromotionRequest` (versioned in
  `contracts/v1/promotion_request.json`).

Schema changes are breaking. Version the contracts. Write them down. Treat
them like an API agreement between two separate teams.

### 24. FastAPI Patterns

- Every endpoint has a Pydantic model for its request body **and** another
  for its response — no raw `dict` bodies.
- Every endpoint lives in a router file under `routers/<resource>.py`.
- **Never** return `200 OK` with `{"error": "..."}` in the body. Raise
  `HTTPException` with the correct status code:

| Code | Use when                                                            |
| ---- | ------------------------------------------------------------------- |
| 200  | Success with body                                                   |
| 201  | Created a new resource                                              |
| 400  | Client sent malformed data                                          |
| 401  | Authentication missing or invalid                                   |
| 403  | Authenticated but not permitted (wrong API key for promotion)        |
| 404  | Resource does not exist                                              |
| 422  | Well-formed but semantically invalid (Pydantic returns automatically) |
| 500  | Unhandled server-side error (generic message — no traces leaked)    |

- Bad inputs return structured errors, never stack traces.
- The promotion endpoint requires an internal API key — it is not user-facing.

---

## Part IV — Toolchain & Hygiene

### 25. Python Code Style

Toolchain: **Black** (line length 88) · **isort** (`profile = "black"`) ·
**flake8** (max 88) · **mypy --strict**.

- 4 spaces, never tabs.
- Double quotes for strings.
- Trailing commas in all multi-line structures.
- Type hints on every function signature.
- 2 blank lines between top-level definitions; 1 between methods.

Import order (blank line between groups):

```python
# 1. stdlib
import asyncio
from typing import AsyncIterator

# 2. third-party
import httpx
from fastapi import APIRouter, Depends

# 3. local
from app.core.settings import get_settings
from app.deps.db import get_session
```

### 26. Naming Conventions

| Element                       | Convention             | Example                   |
| ----------------------------- | ---------------------- | ------------------------- |
| Variables, functions, modules | `snake_case`           | `fetch_drift_report()`    |
| Classes                       | `PascalCase`           | `DriftDetector`           |
| Constants                     | `UPPER_SNAKE_CASE`     | `DRIFT_PSI_THRESHOLD`     |
| Private attributes            | `_leading_underscore`  | `self._client`            |
| Booleans                      | reads as a question    | `is_drifted`, `has_approval` |
| Collections                   | plural                 | `predictions`, `jobs`     |

Functions start with a verb: `get_`, `fetch_`, `load_`, `train_`, `save_`,
`validate_`, `process_`, `build_`, `run_`. No single-letter names except loop
counters (`i`, `j`) and lambdas.

### 27. Security — CRITICAL

- **Never** hardcode keys, tokens, passwords, or connection strings.
- All secrets flow through `Settings` (one place). No `os.getenv(...)`
  scattered across files.
- `.env` is in `.gitignore`. Commit `.env.example` with every required key
  and **fake placeholder values only**.
- If a secret is ever committed, **rotate it immediately**. Removing the
  commit from history is not enough — it persists in forks, clones, and CI
  logs.
- Required secrets: `GOOGLE_API_KEY`, `POSTGRES_PASSWORD`, `PROMOTION_API_KEY`.
  These live in `.env` (gitignored). For production, migrate to HashiCorp
  Vault — see `DECISIONS.md`.
- The promotion endpoint is protected by an internal API key (`PROMOTION_API_KEY`)
  that only the agent worker knows.
- Validate all user input at API boundaries with Pydantic.

### 28. Documentation (Google Style)

Every public module, class, and function has a docstring:

```python
async def compute_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Compute Population Stability Index between two distributions.

    Args:
        reference: The baseline distribution from training data.
        current: The rolling-window distribution from recent predictions.
        bins: Number of quantile bins for the PSI calculation.

    Returns:
        PSI value. < 0.1 = no significant change, 0.1–0.25 = moderate, >= 0.25 = significant.

    Raises:
        ValueError: If either series is empty.
    """
```

Inline comments explain **why**, not **what**. Module docstrings describe
intended functionality and how the module interacts with neighbours.

### 29. Dependency Management with `uv`

This project uses `uv`. Do **not** use `pip` or `venv` directly.

```bash
uv sync                        # install prod + dev deps
uv sync --no-dev               # prod only (production image)
uv run pytest                  # run tests in the venv
uv run uvicorn app.main:app    # start FastAPI
uv add <package>               # add a prod dependency
uv add --dev <package>         # add a dev dependency
```

`pyproject.toml` is the single source of truth. **Pinned exact versions**.
Never use `requirements.txt`. Always commit `uv.lock` — reproducible builds
depend on it.

```toml
[project]
dependencies = [...]           # production only

[dependency-groups]
dev = [...]                    # pytest, black, mypy — never in prod image
```

### 30. Pre-commit Pipeline

```
black → isort → flake8 → mypy → pytest → gitleaks
```

Configured in `.pre-commit-config.yaml`. Run `uv run pre-commit install` once
after cloning. Hooks must pass before any commit lands.

### 31. Git Hygiene

**Branch naming:** `<type>/<short-description>` — lowercase, hyphens, 2–4 words.

| Prefix      | Use for                |
| ----------- | ---------------------- |
| `feature/`  | New functionality      |
| `bugfix/`   | Bug fix                |
| `hotfix/`   | Urgent production fix  |
| `refactor/` | Code restructuring     |
| `docs/`     | Documentation only     |
| `test/`     | Tests added or updated |
| `chore/`    | Maintenance / tooling  |

Never commit directly to `main`.

**Commit messages — Conventional Commits:** `<type>(<scope>): <summary>`.
Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `perf`,
`security`. Imperative mood, capitalise first letter, no trailing period,
≤ 72 chars. Example: `feat(drift): add PSI computation with rolling window`.

**PR rules:** title `[TYPE] Short imperative description`. ≤ 400 lines per PR.
**One concern per PR.** Required body sections: Summary · Changes · Testing ·
Checklist (style, self-review, tests, docs, no secrets, no lint errors).

### 32. Docker & Deployment

- **One service, one container, one Dockerfile.** Model service, agent, worker,
  and dashboard are separate images.
- `docker-compose.yaml` orchestrates the whole stack: postgres, redis, mlflow,
  model-service, agent, worker, dashboard. All come up with `docker-compose up`.
- Services reference each other by service name (`http://service:8000`,
  `http://agent:8001`, `redis://redis:6379`) — never hardcoded IPs.
- Named volumes for `postgres_data`, `redis_data`, and `mlflow_data` — all
  survive restarts so the demo doesn't retrain on every `up`.
- Healthchecks on postgres, redis, model-service, agent.
  `depends_on: condition: service_healthy` so no service starts before its
  dependencies are ready.
- Multi-stage Dockerfiles, slim base images, non-root user.
- **The full stack comes up with `docker-compose up` from a clean clone after
  `cp .env.example .env` and filling in the required secrets.**
- Secrets come from `.env` (gitignored). For production, migrate to
  HashiCorp Vault — see `DECISIONS.md`.

### 33. Testing Requirements

**CI must pass on every push. Refuse to merge if any test regresses.**

Required test categories:

1. **Schema tests** — valid and invalid inputs for every Pydantic model.
2. **Drift computation tests** — known distributions produce known PSI and
   chi² values.
3. **Prediction endpoint tests** — valid inputs return predictions; invalid
   inputs return structured errors (never stack traces).
4. **Promotion gate tests** — the checklist assertions are correct
   (insufficient model metrics → rejection, etc.).
5. **Agent snapshot trajectory tests** — mock the LLM to return predetermined
   tool calls; inject a drift webhook; assert the agent follows the exact
   path through triage → action → comms. Snapshot the full state at each
   step. Any routing change = test failure. **These run without an API key.**
6. **Worker idempotency tests** — enqueue same job twice with same
   idempotency key; assert it executes exactly once.
7. **Fidelity replay test** — load the trained model, predict on a fixed test
   input, assert output matches to 1e-12 precision.

CI pipeline (`infra/github-actions/ci.yml`): build images → run agent
snapshot trajectory tests with mocked LLM → run fidelity replay test → run
all pytest → run linters.

### 34. README Requirements

The README is the front door. A reader should clone, set up, and run the
whole project without asking a single question.

**Required sections:**

1. Project name + one-paragraph description.
2. Architecture overview (with a diagram) — model service, agent, worker,
   dashboard, Redis, Postgres, MLflow.
3. Prerequisites — Docker, `uv`.
4. Setup — clone, `cp .env.example .env`, fill in the required secrets
   (`GOOGLE_API_KEY`, `POSTGRES_PASSWORD`, `PROMOTION_API_KEY`).
5. How to run — `docker compose up` must be one of the commands.
6. Environment variables — every variable, purpose, required/optional.
   All secrets come from `.env` via `Settings`; `.env` is gitignored.
7. **ML narrative:**
   - Feature justifications (why `duration` was dropped, why `pdays==999` is a
     sentinel, why `unknown` is kept as a category).
   - Model comparison table (3 classifiers + baseline, accuracy ± std,
     macro-F1 ± std, per-class, AUC).
   - Threshold tuning rationale (recall >= 0.75 rule).
   - Drift detection methodology (PSI, chi², output distribution).
8. **Agent narrative:**
   - Supervisor topology (triage → action → comms).
   - HIL flow diagram.
   - Queue architecture (arq, idempotency keys, DLQ).
   - How staleness guard works.
9. **Demo script** — step-by-step for the Friday presentation.
10. Deployment notes.

**Required documents alongside README:**

- `ARCH.md` — Architecture diagram and data flow.
- `DECISIONS.md` — Key decisions and tradeoffs (webhook push vs poll, Gemini
  + Ollama fallback, arq vs Celery, HashiCorp Vault for secrets, etc.).
- `RUNBOOK.md` — How to operate, common failure modes, recovery procedures.

**Do not include:** a full library list (that's `pyproject.toml`), a wall of
screenshots, or a full API reference (link to `/docs`).

---

## Part V — Execution & Process

### 35. Execution Roadmap (5 days, Thursday-midnight deadline)

Stage gates — do not advance if the gate is red.

**Day 1 (Mon) — Foundation + ML pipeline.**
Repo init (uv, pre-commit, .env.example with required secrets,
.gitignore, .dockerignore). `docker-compose.yaml` for postgres + redis +
mlflow with healthchecks. `Settings` class with pydantic-settings.
ML training pipeline (`ml/data.py`, `ml/pipeline.py`,
`ml/train.py`, `ml/threshold.py`, `ml/register.py`, `ml/reference_stats.py`).
**Gate:** `uv run python -m drift_triage.ml.train` registers a model in
MLflow; `mlflow ui` shows it; reference-stats JSON saved under `artifacts/`.

**Day 2 (Tue) — Model service, drift, webhook, promotion gate.**
FastAPI service with lifespan-loaded classifier and ref stats.
`/predict`, `/drift/report`, `/promotion/promote` endpoints. PSI / chi² /
output-PSI implementations. Severity-change detector emits webhook in
`BackgroundTasks` with tenacity retries. Promotion gate asserts the day-4
checklist behind `X-Promotion-Key`. `contracts/v1/drift_webhook.json`
versioned. Tests for schemas, drift math, prediction errors, promotion
rejection paths.
**Gate:** `docker compose up`; `curl /predict` works; injecting drift into
the rolling-window table flips `/drift/report` severity to high; webhook
hits a stub agent endpoint and is logged.

**Day 3 (Wed) — Agent + queue.**
`BaseTool` ABC with `safe_run()`. Tools: `inspect_drift`, `replay_test`,
`retrain`, `rollback`, `propose_action`, `request_hil_approval`,
`dispatch_action`, `compose_summary`, `update_dashboard`. Gemini primary +
Ollama fallback (`@lru_cache(maxsize=1)`, `response_schema`,
`max_output_tokens`, tenacity). LangGraph supervisor topology:
`webhook → triage → action → (HIL?) → comms → dispatch → END`. Postgres
checkpointer. arq worker with `SETNX`-based idempotency, retries 1s→2s→4s,
DLQ at `drift_actions:dlq`.
**Gate:** Inject webhook → investigation opens → triage → action → HIL pause
persists → kill agent container → restart → investigation resumes from
checkpoint.

**Day 4 AM (Thu) — Dashboard, CI, docs, end-to-end.**
Streamlit dashboard with auto-refresh 5s (registry view, drift report, open
+ resolved investigations, queue/DLQ, HIL inbox). CI workflow: build images,
lint, pytest with coverage gates, snapshot trajectory test (no API key),
1e-12 fidelity replay. Docs: `README.md`, `ARCH.md`, `DECISIONS.md`,
`RUNBOOK.md`, `BUGS.md`.
**Gate:** Fresh clone on a second machine: `cp .env.example .env`, fill
three vars, `docker compose up`, run the demo script end-to-end. CI green.

**Day 4 PM (Thu) — Tag + submit + rehearse.**
`git tag v0.1.0-week5 && git push --tags`. Submission message per the
brief's format. Rehearse the demo twice with a 5-minute timer.

If a stage gate is red at end of day, the next day's first task is fixing
it — not starting the new day's work.

### 36. Demo & Submission Discipline

**The Friday demo is 5 minutes. Plan to the second.**

| Slot | Time | Content                                                                                       |
| ---- | ---- | --------------------------------------------------------------------------------------------- |
| 1    | 60s  | Architecture walkthrough — diagram on screen, point at boxes.                                 |
| 2    | 180s | Live: shift `euribor3m` +2σ + shift `job` distribution → drift detected → webhook → investigation opens → triage → action proposes retrain → HIL approve → arq dispatches → dashboard reflects. |
| 3    | 30s  | Show one CI failure on a snapshot trajectory regression (intentional break on a branch).      |
| 4    | 30s  | Show one specific real bug you hit and fixed.                                                 |

**`BUGS.md` is populated during development, not at the end.** Every real
bug that costs you more than 30 minutes gets a one-paragraph entry: symptom,
root cause, fix, lesson. The Friday "one bug" slot pulls from this file —
it is **not** invented retrospectively.

**Submission message** (paste exactly into the submission channel):

```
Project 5 - [Name 1] | [Name 2]
Repo: [GitHub URL]
Tag: v0.1.0-week5
Dataset: UCI Bank Marketing (bank-additional-full.csv)
Model: [registered name + version] (Test AUC: [n] | Test F1: [n])
Operating threshold: [n] (rule: recall >= 0.75)
LLM: Gemini 2.5 Flash + Ollama fallback - chosen because [one line]
README contains: ARCH.md, DECISIONS.md, RUNBOOK.md
```

Tag command: `git tag -a v0.1.0-week5 -m "Week 5 submission" && git push --tags`.

The repo must come up cleanly on a fresh clone after `cp .env.example .env`
and filling the required secrets (`GOOGLE_API_KEY`, `POSTGRES_PASSWORD`,
`PROMOTION_API_KEY`). Test this on a different
machine (or a clean Docker volume set) before submitting. "It works on my
laptop" is the failure mode this gate exists to prevent.

### 37. Pair-Working Norms

**Two students. Both names on the repo. Both names answer on Friday.**

- **Both review every PR.** Partner is the required reviewer; no PR merges
  without their approval. `CODEOWNERS` lists both names for the whole tree.
- **Either of you must be able to explain any line.** The brief states one of
  you will be asked to explain the *other's* code on Friday. Pair regularly;
  do not silo (one on ML, one on agent) without a daily 15-minute walkthrough
  of what the other shipped.
- **Conventional Commits with the author who wrote it.** No
  `Co-authored-by:` flags as a substitute for understanding — those are
  fine, but the listed author is responsible for the line.
- **Daily sync on the stage gate.** End of each day, both partners walk
  through the gate together and confirm green. If red, both partners stop
  and fix it before starting the next day's work.
- **Branch naming includes initials when useful** (e.g.
  `feature/am-drift-psi`, `feature/sb-agent-graph`) so the partner can
  spot whose branch is whose at a glance. Optional but reduces friction.
- **No "I did the easy half."** Friday questions are random; if one of you
  cannot defend the agent loop and the other cannot defend the drift math,
  the demo fails for both.

---

## Key Decisions

| Decision                  | Choice                                          | Rationale                                                         |
| ------------------------- | ----------------------------------------------- | ----------------------------------------------------------------- |
| LLM provider              | **Gemini** (primary) + **Ollama** (fallback)    | Gemini for quality; Ollama for resilience when API is down       |
| Webhook vs poll           | **Push** (platform → agent)                    | Lower latency; simpler state model; no polling interval to tune   |
| Queue library             | **arq** (async, Redis-backed)                   | Lightweight, async-native, built-in retry/DLQ, Redis-backed       |
| Agent checkpoint store    | **Postgres** via `langgraph-checkpoint-postgres` | Specified in brief; survives restarts; queryable                  |
| Model storage             | **MLflow** with Postgres backend store           | Persisted across restarts; full model registry; versioning         |
| Streamlit refresh         | **Auto-poll every 5 seconds**                   | Demo requires seeing live updates                                 |
| Secrets management        | **`.env` + `pydantic-settings`** (Vault for production) | Simplicity for bootcamp; `extra="forbid"` catches typos; Vault migration path in `DECISIONS.md` |

---

## Pre-Review Checklist

Run through this before every PR. If you can't answer "yes" to all of it,
you have work to do.

- [ ] I can explain what every file does and why it is named that way.
- [ ] Every route, tool, and external call is async. No `requests`, no
      `time.sleep`, no blocking I/O in the request path.
- [ ] Every dependency (DB session, classifier, LLM client, Redis pool) is
      declared with `Depends()`. No globals.
- [ ] Heavy resources (engine, classifier, Redis pool, LLM clients) load once
      in lifespan and dispose on shutdown.
- [ ] `lru_cache` on deterministic helpers; TTL cache on at least one external
      call where it makes sense; thundering-herd lock present where needed.
- [ ] All config goes through `Settings`. No `os.getenv` outside it.
      `extra="forbid"` is set. Secrets come from `.env`; `.env` is gitignored.
- [ ] Every external boundary (HTTP req/resp, tool input/output, LLM
      structured output, webhook payload) has a Pydantic model.
- [ ] Every external call has a timeout, tenacity retries with backoff
      (transient only), and structured error returns from tools.
- [ ] Code is split into modules by concern. Logging is structured. Linter and
      formatter run on every commit.
- [ ] Pydantic schemas, tool logic, agent snapshot trajectories, and fidelity
      replay are tested. Tests run in CI.
- [ ] Every endpoint lives in an `APIRouter`, raises `HTTPException` with the
      correct status code.
- [ ] LLM calls use `response_schema` with Pydantic; system prompt and user
      prompt are separated; every call has `max_output_tokens`.
- [ ] LLM fallback to Ollama is implemented and tested.
- [ ] Prompt-injection guardrails (sanitize, delimit, log) are applied
      everywhere external data reaches a prompt.
- [ ] `duration` column is dropped. `pdays==999` is flagged. `unknown` is a
      real category. `random_state=42` on every stochastic call.
- [ ] The promotion gate asserts the day-4 checklist — no version reaches
      Production without it.
- [ ] HIL approval pauses the agent before Production changes, and a staleness
      guard aborts if a newer drift event has arrived.
- [ ] Queue jobs have idempotency keys. Duplicate dispatches are deduplicated.
      Failed jobs go to a DLQ after max retries.
- [ ] Agent state persists in Postgres checkpoints. Kill and restart resumes
      the last investigation, not a new one.
- [ ] The contract between platform and agent is versioned in `contracts/v1/`.
- [ ] `.env`, `.venv`, model artefacts, and large data files are in
      `.gitignore`. No secrets in git. Secrets loaded from `.env` via one
      `Settings` class.
- [ ] Each service has its own Dockerfile; `docker compose up` runs the whole
      stack from a clean machine.
- [ ] `uv` used for environments; `uv.lock` committed.
- [ ] No AI-generated code I cannot explain line by line.

---

*The reason production codebases feel different from tutorial code is not that
they use fancier libraries. It's that every part of the codebase respects the
same set of standards, and the parts compose because of it.*