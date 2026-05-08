"""Streamlit dashboard for the Week 5 Drift Triage Co-Pilot demo."""

from __future__ import annotations

import csv
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import streamlit as st

SERVICE_URL = os.getenv("SERVICE_URL", "http://service:8000").rstrip("/")
AGENT_URL = os.getenv("AGENT_URL", "http://agent:8001").rstrip("/")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://mlflow:5000").rstrip("/")
MLFLOW_PUBLIC_URL = os.getenv("MLFLOW_PUBLIC_URL", "http://localhost:5001")

SCENARIOS = {
    "normal": {
        "button": "Send Normal Batch",
        "file": "normal_2000.csv",
        "expected_action": "no action / monitor",
    },
    "replay_drift": {
        "button": "Send Replay Drift Batch",
        "file": "replay_drift_2000.csv",
        "expected_action": "replay_test",
    },
    "retrain_drift": {
        "button": "Send Retrain Drift Batch",
        "file": "retrain_drift_2000.csv",
        "expected_action": "retrain",
    },
    "rollback_drift": {
        "button": "Send Rollback Drift Batch",
        "file": "rollback_drift_2000.csv",
        "expected_action": "rollback or production action",
    },
}

PREDICTION_FIELDS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]
INTEGER_FIELDS = {"age", "campaign", "pdays", "previous"}

st.set_page_config(
    page_title="Drift Triage Co-Pilot",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@dataclass
class ApiResult:
    ok: bool
    data: Any = None
    error: str | None = None
    status_code: int | None = None


def api_get(url: str, timeout: float = 5.0) -> ApiResult:
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url)
            response.raise_for_status()
            try:
                data = response.json()
            except ValueError:
                data = response.text
            return ApiResult(ok=True, data=data, status_code=response.status_code)
    except httpx.HTTPStatusError as exc:
        return ApiResult(
            ok=False,
            error=f"HTTP {exc.response.status_code}: {exc.response.text}",
            status_code=exc.response.status_code,
        )
    except Exception as exc:
        return ApiResult(ok=False, error=str(exc))


def api_post(
    url: str,
    payload: dict[str, Any],
    timeout: float = 10.0,
    headers: dict[str, str] | None = None,
) -> ApiResult:
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=payload, headers=headers or {})
            response.raise_for_status()
            return ApiResult(
                ok=True, data=response.json(), status_code=response.status_code
            )
    except httpx.HTTPStatusError as exc:
        return ApiResult(
            ok=False,
            error=f"HTTP {exc.response.status_code}: {exc.response.text}",
            status_code=exc.response.status_code,
        )
    except Exception as exc:
        return ApiResult(ok=False, error=str(exc))


def badge(label: str, state: str) -> str:
    colors = {
        "healthy": ("#047857", "#d1fae5"),
        "warning": ("#b45309", "#fef3c7"),
        "error": ("#b91c1c", "#fee2e2"),
        "not available": ("#475569", "#e2e8f0"),
        "high": ("#b91c1c", "#fee2e2"),
        "medium": ("#b45309", "#fef3c7"),
        "low": ("#047857", "#d1fae5"),
        "none": ("#475569", "#e2e8f0"),
        "open": ("#1d4ed8", "#dbeafe"),
        "resolved": ("#047857", "#d1fae5"),
        "pending": ("#b45309", "#fef3c7"),
    }
    fg, bg = colors.get(state.lower(), colors["not available"])
    return (
        f"<span class='badge' style='color:{fg};background:{bg};"
        f"border-color:{fg}22'>{label}</span>"
    )


def render_card(title: str, value: str, status: str, help_text: str = "") -> None:
    st.markdown(
        "<div class='card'>"
        f"<div class='card-title'>{title}</div>"
        f"<div class='card-value'>{value}</div>"
        f"<div>{badge(status.title(), status)}</div>"
        f"<div class='card-help'>{help_text}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def json_file(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def demo_batch_dir() -> Path | None:
    env_dir = os.getenv("DEMO_BATCH_DIR")
    candidates = [
        Path(env_dir).expanduser() if env_dir else None,
        project_root() / "data/demo_batches",
        Path(__file__).resolve().parent / "data/demo_batches",
        Path("/app/data/demo_batches"),
        Path("/data/demo_batches"),
    ]
    return first_existing([p for p in candidates if p is not None])


def artifact_reports_dir() -> Path | None:
    env_dir = os.getenv("ARTIFACT_REPORT_DIR")
    candidates = [
        Path(env_dir).expanduser() if env_dir else None,
        project_root() / "backend/artifacts/reports",
        Path("/app/artifacts/reports"),
        Path("/artifacts/reports"),
    ]
    return first_existing([p for p in candidates if p is not None])


def batch_path(scenario: str) -> Path | None:
    base = demo_batch_dir()
    if not base:
        return None
    path = base / SCENARIOS[scenario]["file"]
    return path if path.exists() else None


def read_batch_rows(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def prediction_payload(row: dict[str, str]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field in PREDICTION_FIELDS:
        if field not in row or row[field] in ("", None):
            raise ValueError(f"Missing prediction field: {field}")
        if field in INTEGER_FIELDS:
            payload[field] = int(float(row[field]))
        elif field in {
            "emp.var.rate",
            "cons.price.idx",
            "cons.conf.idx",
            "euribor3m",
            "nr.employed",
        }:
            payload[field] = float(row[field])
        else:
            payload[field] = str(row[field])
    return payload


def force_drift_report() -> ApiResult:
    return api_get(f"{SERVICE_URL}/api/v1/drift/report?force=true", timeout=30.0)


def reset_drift_window() -> ApiResult:
    return api_post(f"{SERVICE_URL}/api/v1/drift/reset", {}, timeout=10.0)


def top_drift_features(report: dict[str, Any], limit: int = 3) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for result in report.get("psi_results") or []:
        psi = float(result.get("psi") or 0)
        items.append(
            {
                "feature": result.get("feature"),
                "type": "PSI",
                "value": round(psi, 4),
                "severity": result.get("severity", "unknown"),
                "_sort": (
                    {"high": 3, "medium": 2, "low": 1}.get(
                        str(result.get("severity")), 0
                    ),
                    psi,
                ),
            }
        )
    for result in report.get("chi2_results") or []:
        p_value = float(result.get("p_value") or 1)
        items.append(
            {
                "feature": result.get("feature"),
                "type": "Chi-square",
                "value": round(p_value, 6),
                "severity": result.get("severity", "unknown"),
                "_sort": (
                    {"high": 3, "medium": 2, "low": 1}.get(
                        str(result.get("severity")), 0
                    ),
                    1 - p_value,
                ),
            }
        )
    sorted_items = sorted(items, key=lambda x: x["_sort"], reverse=True)[:limit]
    return [{k: v for k, v in item.items() if k != "_sort"} for item in sorted_items]


def inject_demo_batch(scenario: str, max_rows: int | None = None) -> dict[str, Any]:
    path = batch_path(scenario)
    if path is None:
        raise FileNotFoundError(
            "Demo batch not found. Mount data/demo_batches into the dashboard or run "
            "scripts/generate_demo_batches.py first."
        )

    rows = read_batch_rows(path, limit=max_rows)
    started = time.time()
    success_count = 0
    failure_count = 0
    error_examples: list[dict[str, Any]] = []
    progress = st.progress(0, text=f"Sending {len(rows)} rows through /api/v1/predict")

    for index, row in enumerate(rows, start=1):
        try:
            payload = prediction_payload(row)
            result = api_post(f"{SERVICE_URL}/api/v1/predict", payload, timeout=10.0)
            if result.ok:
                success_count += 1
            else:
                failure_count += 1
                if len(error_examples) < 5:
                    error_examples.append({"row": index, "error": result.error})
        except Exception as exc:
            failure_count += 1
            if len(error_examples) < 5:
                error_examples.append({"row": index, "error": str(exc)})

        if index == len(rows) or index % 25 == 0:
            progress.progress(
                index / max(len(rows), 1), text=f"Sent {index}/{len(rows)} rows"
            )

    progress.empty()
    report_result = (
        force_drift_report()
        if success_count
        else ApiResult(ok=False, error="No successful predictions")
    )
    report = report_result.data if report_result.ok else {}
    return {
        "scenario": scenario,
        "rows_requested": len(rows),
        "success_count": success_count,
        "failure_count": failure_count,
        "elapsed_seconds": round(time.time() - started, 1),
        "expected_action": SCENARIOS[scenario]["expected_action"],
        "drift_severity": report.get("severity", "not available"),
        "top_drifted_features": top_drift_features(report),
        "errors": error_examples,
        "drift_error": None if report_result.ok else report_result.error,
    }


def extract_field(summary: str | None, name: str) -> str:
    if not summary:
        return "Not available"
    match = re.search(rf"{re.escape(name)}=([^,\.\s]+)", summary)
    return match.group(1) if match else "Not available"


def format_time(value: Any) -> str:
    return str(value) if value else "Not available"


def model_artifact_summary() -> dict[str, Any]:
    reports_dir = artifact_reports_dir()
    if not reports_dir:
        return {}
    threshold = json_file(reports_dir / "operating_threshold.json") or {}
    training = json_file(reports_dir / "training_report.json") or {}
    return {
        "registered_model_name": threshold.get(
            "registered_model_name", "drift-triage-classifier"
        ),
        "selected_model": threshold.get("selected_model")
        or training.get("selected_model"),
        "operating_threshold": threshold.get("operating_threshold")
        or training.get("selected_threshold"),
        "created_at": threshold.get("created_at") or training.get("created_at"),
        "test_metrics": training.get("test_metrics") or {},
    }


st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 3rem;}
    .card {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        background: #ffffff;
        min-height: 126px;
    }
    .card-title {
        font-size: 0.78rem; color: #64748b;
        text-transform: uppercase; letter-spacing: 0;
    }
    .card-value {
        font-size: 1.15rem; font-weight: 650;
        color: #0f172a; margin: 0.25rem 0 0.45rem;
    }
    .card-help {font-size: 0.82rem; color: #64748b; margin-top: 0.4rem;}
    .badge {
        border: 1px solid;
        border-radius: 999px;
        padding: 0.18rem 0.55rem;
        font-size: 0.78rem;
        font-weight: 650;
        display: inline-block;
    }
    .small-note {color: #64748b; font-size: 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Drift Triage Co-Pilot")

if "last_batch_result" not in st.session_state:
    st.session_state.last_batch_result = None

service_health = api_get(f"{SERVICE_URL}/health")
agent_health = api_get(f"{AGENT_URL}/health")
mlflow_health = api_get(f"{MLFLOW_URL}/health")
drift_result = (
    force_drift_report()
    if service_health.ok
    else ApiResult(ok=False, error="Service unavailable")
)
drift_report = drift_result.data if drift_result.ok else {}
model_summary = model_artifact_summary()

st.subheader("Demo Control Panel")
health_cols = st.columns(5)
with health_cols[0]:
    render_card(
        "Model Service",
        "Online" if service_health.ok else "Unavailable",
        "healthy" if service_health.ok else "error",
        SERVICE_URL,
    )
with health_cols[1]:
    render_card(
        "Agent",
        "Online" if agent_health.ok else "Unavailable",
        "healthy" if agent_health.ok else "error",
        AGENT_URL,
    )
with health_cols[2]:
    render_card(
        "MLflow",
        "Open UI",
        "healthy" if mlflow_health.ok else "warning",
        f"[{MLFLOW_PUBLIC_URL}]({MLFLOW_PUBLIC_URL})",
    )
with health_cols[3]:
    threshold = model_summary.get("operating_threshold")
    render_card(
        "Current Model",
        model_summary.get("registered_model_name", "Not available"),
        "healthy" if service_health.ok else "warning",
        (
            f"Threshold {threshold:.3f}"
            if isinstance(threshold, (int, float))
            else "Model/MLflow unavailable — train or start MLflow first."
        ),
    )
with health_cols[4]:
    severity = drift_report.get("severity", "not available")
    render_card(
        "Drift Severity",
        str(severity).title(),
        str(severity),
        f"Window size {drift_report.get('window_size', 'not available')}",
    )

if not service_health.ok:
    st.warning(
        "Model/MLflow unavailable - train or start MLflow before prediction demo."
    )
if not agent_health.ok:
    st.warning(
        "Agent is unavailable. Batch prediction and drift viewing may still work,"
        " but investigations and HIL approval will not update."
    )

st.divider()
st.subheader("Demo Batch Launcher")
batch_dir = demo_batch_dir()
if batch_dir:
    st.caption(f"Demo batches: {batch_dir}")
else:
    st.warning(
        "Demo batch directory not available. "
        "Expected data/demo_batches with generated CSV files."
    )

batch_row_count = st.number_input(
    "Rows to inject",
    min_value=50,
    max_value=10000,
    value=200,
    step=50,
    help="How many rows from the batch CSV to send through the prediction API.",
)

button_cols = st.columns(5)
for index, (scenario, info) in enumerate(SCENARIOS.items()):
    with button_cols[index]:
        available = batch_path(scenario) is not None and service_health.ok
        if st.button(
            info["button"],
            key=f"send_{scenario}",
            disabled=not available,
            use_container_width=True,
        ):
            with st.spinner(f"Sending {scenario} through the prediction API..."):
                try:
                    st.session_state.last_batch_result = inject_demo_batch(
                        scenario, max_rows=int(batch_row_count)
                    )
                    st.success("Batch sent. Drift report refreshed.")
                    st.rerun()
                except Exception as exc:
                    st.session_state.last_batch_result = {
                        "scenario": scenario,
                        "errors": [{"row": "setup", "error": str(exc)}],
                    }
                    st.error(str(exc))
with button_cols[4]:
    st.markdown("&nbsp;", unsafe_allow_html=True)
    if st.button(
        "Reset Demo",
        key="reset_demo",
        use_container_width=True,
        type="secondary",
        disabled=not service_health.ok,
    ):
        r1 = reset_drift_window()
        r2 = api_post(f"{AGENT_URL}/admin/reset", {}, timeout=10.0)
        if r1.ok and r2.ok:
            st.session_state.last_batch_result = None
            st.success(
                "Reset complete. Send Normal Batch → then a drift batch "
                "to trigger a fresh investigation."
            )
            st.rerun()
        else:
            st.error(f"Reset errors: service={r1.error}, agent={r2.error}")

if st.session_state.last_batch_result:
    result = st.session_state.last_batch_result
    st.markdown("**Last batch result**")
    metrics = st.columns(6)
    metrics[0].metric("Rows Requested", result.get("rows_requested", "n/a"))
    metrics[1].metric("Success", result.get("success_count", "n/a"))
    metrics[2].metric("Failures", result.get("failure_count", "n/a"))
    metrics[3].metric("Elapsed Seconds", result.get("elapsed_seconds", "n/a"))
    metrics[4].metric("Expected Action", result.get("expected_action", "n/a"))
    metrics[5].metric("Drift Severity", result.get("drift_severity", "n/a"))
    if result.get("top_drifted_features"):
        st.dataframe(
            result["top_drifted_features"], use_container_width=True, hide_index=True
        )
    if result.get("drift_error"):
        st.error(f"Drift report failed: {result['drift_error']}")
    if result.get("errors"):
        with st.expander("Batch error output"):
            st.json(result["errors"])

st.divider()
inv_left, inv_right = st.columns([3, 1])
with inv_left:
    st.subheader("Agent Investigations")
with inv_right:
    st.markdown("&nbsp;", unsafe_allow_html=True)
    if st.button("Refresh Investigations", use_container_width=True, key="refresh_inv"):
        st.rerun()
investigations_result = api_get(f"{AGENT_URL}/investigations")
investigations = investigations_result.data if investigations_result.ok else []
if not investigations_result.ok:
    st.error(f"Could not load investigations: {investigations_result.error}")
elif not investigations:
    st.info("No investigations yet. Send a drifted batch to trigger the agent.")
else:
    open_items = [item for item in investigations if item.get("status") != "resolved"]
    resolved_items = [
        item for item in investigations if item.get("status") == "resolved"
    ]
    tabs = st.tabs([f"Open ({len(open_items)})", f"Resolved ({len(resolved_items)})"])
    for tab, items in zip(tabs, [open_items, resolved_items], strict=True):
        with tab:
            for item in items:
                summary = item.get("summary_md") or ""
                severity = extract_field(summary, "severity")
                action = extract_field(summary, "action")
                label = f"{item.get('id', '')[:8]} · {severity} · {action}"
                with st.expander(label):
                    inv_cols = st.columns(4)
                    inv_cols[0].write(f"**Severity**: {severity}")
                    inv_cols[1].write(
                        f"**Model**: {extract_field(summary, 'model_name')}"
                        f" v{extract_field(summary, 'model_version')}"
                    )
                    inv_cols[2].write(f"**Action**: {action}")
                    inv_cols[3].write(
                        f"**Updated**: {format_time(item.get('updated_at'))}"
                    )
                    st.write(summary or "No summary available yet.")

st.divider()
hil_left, hil_right = st.columns([3, 1])
with hil_left:
    st.subheader("Human Approval Inbox")
with hil_right:
    st.markdown("&nbsp;", unsafe_allow_html=True)
    if st.button("Refresh HIL", use_container_width=True, key="refresh_hil"):
        st.rerun()
approvals_result = api_get(f"{AGENT_URL}/hil/approvals")
approvals = approvals_result.data if approvals_result.ok else []
if not approvals_result.ok:
    st.error(f"Could not load HIL approvals: {approvals_result.error}")
elif not approvals:
    st.success("No pending approvals.")
else:
    for approval in approvals:
        approval_id = approval.get("id")
        investigation_id = approval.get("investigation_id")
        with st.container(border=True):
            st.markdown(f"**{approval.get('action', 'action')} approval required**")
            cols = st.columns(3)
            cols[0].write(f"**Related investigation**: `{str(investigation_id)[:8]}`")
            cols[1].write(f"**Model version**: {approval.get('model_version', 'n/a')}")
            cols[2].write(f"**Created**: {format_time(approval.get('created_at'))}")
            reason = (
                approval.get("rationale")
                or approval.get("summary_md")
                or "Not available"
            )
            st.write(f"**Reason**: {reason}")
            st.write(
                "**Risk / impact**: This action may enqueue worker activity "
                "that touches replay, retraining, or production model state."
            )
            approve_col, reject_col, _ = st.columns([1, 1, 4])
            with approve_col:
                if st.button(
                    "Approve", key=f"approve_{approval_id}", use_container_width=True
                ):
                    with st.spinner("Sending approval to agent..."):
                        result = api_post(
                            f"{AGENT_URL}/hil/approve",
                            {
                                "investigation_id": investigation_id,
                                "hil_approval_id": approval_id,
                                "decision": "approved",
                            },
                            timeout=30.0,
                        )
                    if not result.ok:
                        st.error(result.error)
                        st.stop()
                    st.success(
                        "HIL approved — worker has picked up the retrain job. "
                        "Refresh Queue / Registry to track progress."
                    )
                    st.rerun()
            with reject_col:
                if st.button(
                    "Reject", key=f"reject_{approval_id}", use_container_width=True
                ):
                    result = api_post(
                        f"{AGENT_URL}/hil/approve",
                        {
                            "investigation_id": investigation_id,
                            "hil_approval_id": approval_id,
                            "decision": "rejected",
                        },
                        timeout=30.0,
                    )
                    st.info("Rejected.") if result.ok else st.error(result.error)
                    time.sleep(0.5)
                    st.rerun()

st.divider()
qw_left, qw_right = st.columns([3, 1])
with qw_left:
    st.subheader("Queue / Worker")
with qw_right:
    st.markdown("&nbsp;", unsafe_allow_html=True)
    if st.button("Refresh Queue", use_container_width=True, key="refresh_queue"):
        st.rerun()
queue_metrics_result = api_get(f"{AGENT_URL}/queue/metrics")
if queue_metrics_result.ok:
    metrics_data = queue_metrics_result.data or {}
    depth = metrics_data.get("queue_depth", 0)
    dispatches = metrics_data.get("active_dispatches", 0)
    worker_running = metrics_data.get("worker_running", False)
    queue_cols = st.columns(4)
    queue_cols[0].metric("Queue Depth", depth)
    queue_cols[1].metric("DLQ Count", metrics_data.get("dlq_count", 0))
    queue_cols[2].metric("Active Dispatches", dispatches)
    queue_cols[3].metric("Completed Jobs", metrics_data.get("recent_jobs_count", 0))
    if depth > 0:
        st.info(
            f"Job waiting in queue (depth={depth}). Worker will pick it up shortly."
        )
    elif worker_running:
        st.warning(
            "Worker is processing a job (dispatch lock active, queue empty). "
            "Retrain may take several minutes — click Refresh Queue to update."
        )
    dlq_items = metrics_data.get("dlq_items") or []
    if dlq_items:
        with st.expander(f"Dead-letter queue ({len(dlq_items)} items)", expanded=True):
            for item in dlq_items:
                st.error(
                    f"DLQ: {item.get('job_type', 'unknown')} — "
                    f"{item.get('payload', {})}"
                )
else:
    st.warning(f"Queue metrics unavailable: {queue_metrics_result.error}")
    st.caption("Check worker logs with `docker compose logs worker`.")

st.divider()
reg_left, reg_right = st.columns([3, 1])
with reg_left:
    st.subheader("Registry / Model")
with reg_right:
    st.markdown("&nbsp;", unsafe_allow_html=True)
    if st.button("Refresh Registry", use_container_width=True, key="refresh_registry"):
        st.rerun()

registry_result = api_get(f"{SERVICE_URL}/api/v1/models/registry", timeout=5.0)
if registry_result.ok:
    reg_data = registry_result.data or {}
    prod = reg_data.get("production")
    staging = reg_data.get("staging")
    prod_col, staging_col = st.columns(2)

    with prod_col:
        st.markdown("**Production**")
        if prod:
            st.metric("Version", f"v{prod['version']}")
            st.metric("AUC", f"{prod['auc']:.4f}" if prod["auc"] else "n/a")
            st.metric(
                "Recall (val)", f"{prod['recall']:.4f}" if prod["recall"] else "n/a"
            )
            st.metric(
                "Threshold", f"{prod['threshold']:.4f}" if prod["threshold"] else "n/a"
            )
        else:
            st.info("No Production model.")

    with staging_col:
        st.markdown("**Staging**")
        if staging:
            st.metric("Version", f"v{staging['version']}")
            st.metric("AUC", f"{staging['auc']:.4f}" if staging["auc"] else "n/a")
            st.metric(
                "Recall (val)",
                f"{staging['recall']:.4f}" if staging["recall"] else "n/a",
            )
            st.metric(
                "Threshold",
                f"{staging['threshold']:.4f}" if staging["threshold"] else "n/a",
            )
            # Eligibility badge
            recall_ok = staging["recall"] >= 0.75
            auc_ok = (not prod) or (staging["auc"] >= prod["auc"])
            if recall_ok and auc_ok:
                st.success("✓ Eligible for promotion")
            elif not recall_ok:
                st.error(
                    f"✗ Below recall gate "
                    f"({staging['recall']:.4f} < 0.75) — retrying → DLQ"
                )
            else:
                prod_auc = prod["auc"] if prod else 0.0
                st.warning(
                    f"⚠ AUC regression "
                    f"({staging['auc']:.4f} < prod {prod_auc:.4f})"
                    " — keeping Production"
                )
        else:
            st.info("No Staging model.")

    st.caption(
        f"Model: {reg_data.get('model_name', 'unknown')} · "
        f"[Open MLflow UI]({MLFLOW_PUBLIC_URL})"
    )
else:
    st.warning(f"Registry unavailable: {registry_result.error}")
    st.markdown(f"[Open MLflow UI]({MLFLOW_PUBLIC_URL})")

with st.expander("Presentation Demo Steps", expanded=False):
    st.markdown(
        """
        1. Start stack: `make up-full` or `make up`
        2. Send normal batch
        3. Check drift baseline
        4. Send drifted batch
        5. Refresh drift report
        6. Watch investigation open
        7. Approve HIL action
        8. Check queue/worker status
        """
    )
