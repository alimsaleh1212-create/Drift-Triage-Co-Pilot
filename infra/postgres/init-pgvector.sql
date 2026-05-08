-- Enable pgvector extension (harmless if already enabled)
CREATE EXTENSION IF NOT EXISTS vector;

-- Rolling-window prediction log for drift detection
CREATE TABLE IF NOT EXISTS predictions (
    id          TEXT PRIMARY KEY,
    features    JSONB NOT NULL,
    label       SMALLINT NOT NULL,
    probability DOUBLE PRECISION NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS predictions_created_at_idx ON predictions (created_at DESC);

-- Investigation state (agent checkpoints live in langgraph's own tables)
CREATE TABLE IF NOT EXISTS investigations (
    id               TEXT PRIMARY KEY,
    drift_event_id   TEXT,
    drift_report_id  TEXT,
    status           TEXT NOT NULL DEFAULT 'open',
    summary_md       TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- HIL approval requests
CREATE TABLE IF NOT EXISTS hil_approvals (
    id               TEXT PRIMARY KEY,
    investigation_id TEXT NOT NULL REFERENCES investigations(id),
    action           TEXT NOT NULL,
    rationale        TEXT NOT NULL,
    model_version    INTEGER NOT NULL,
    status           TEXT NOT NULL DEFAULT 'pending',
    decision         TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    decided_at       TIMESTAMPTZ
);

-- Promotion audit log
CREATE TABLE IF NOT EXISTS promotion_events (
    id               TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    model_name       TEXT NOT NULL,
    promoted_version INTEGER NOT NULL,
    previous_version INTEGER,
    investigation_id TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
