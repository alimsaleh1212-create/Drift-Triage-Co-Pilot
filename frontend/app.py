"""Streamlit dashboard: registry, drift monitoring, investigations, queue, HIL."""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st
from streamlit_autorefresh import st_autorefresh  # type: ignore[import-untyped]

SERVICE_URL = os.getenv("SERVICE_URL", "http://service:8000")
AGENT_URL = os.getenv("AGENT_URL", "http://agent:8001")

st.set_page_config(
    page_title="Drift Triage Co-Pilot",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Auto-refresh every 5 seconds for live updates
st_autorefresh(interval=5000, key="auto_refresh")


@st.cache_data(ttl=5)
def _get(url: str) -> Any:
    """Fetch JSON from a service endpoint; return None on error."""
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(url)
            r.raise_for_status()
            return r.json()
    except Exception as exc:
        st.error(f"Service unavailable: {exc}")
        return None


def _post(url: str, json: dict[str, Any], headers: dict[str, str] | None = None) -> Any:
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.post(url, json=json, headers=headers or {})
            r.raise_for_status()
            return r.json()
    except Exception as exc:
        st.error(f"Request failed: {exc}")
        return None


# ── Sidebar navigation ─────────────────────────────────────────────────────
page = st.sidebar.radio(
    "Navigate",
    ["Drift Monitor", "Investigations", "Queue", "HIL Inbox", "Registry"],
)

# ── Drift Monitor ───────────────────────────────────────────────────────────
if page == "Drift Monitor":
    st.title("Drift Monitor")
    report = _get(f"{SERVICE_URL}/api/v1/drift/report")

    if report:
        severity = report.get("severity", "unknown")
        color = {"low": "green", "medium": "orange", "high": "red"}.get(severity, "gray")
        st.markdown(
            f"**Overall severity:** :{color}[{severity.upper()}]  |  "
            f"Window: {report.get('window_size', 0)} predictions"
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("PSI — Numeric Features")
            psi_results = report.get("psi_results", [])
            if psi_results:
                import plotly.graph_objects as go

                features = [r["feature"] for r in psi_results]
                psi_values = [r["psi"] for r in psi_results]
                bar_colors = [
                    "red" if r["severity"] == "high"
                    else "orange" if r["severity"] == "medium"
                    else "green"
                    for r in psi_results
                ]
                fig = go.Figure(
                    go.Bar(x=features, y=psi_values, marker_color=bar_colors)
                )
                fig.add_hline(y=0.1, line_dash="dash", line_color="orange")
                fig.add_hline(y=0.25, line_dash="dash", line_color="red")
                fig.update_layout(xaxis_tickangle=-45, height=300, margin=dict(t=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric feature results yet.")

        with col2:
            st.subheader("Chi² — Categorical Features")
            chi2_results = report.get("chi2_results", [])
            if chi2_results:
                for r in chi2_results:
                    badge = "🔴" if r["severity"] == "high" else "🟠" if r["severity"] == "medium" else "🟢"
                    st.write(f"{badge} `{r['feature']}` — p={r['p_value']:.4f}")
            else:
                st.info("No categorical results yet.")

        st.subheader("Output Distribution Drift")
        od = report.get("output_drift", {})
        if od:
            st.metric("Output PSI", f"{od.get('psi', 0):.4f}", delta=None)
            st.write(
                f"Reference class-1 rate: {od.get('reference_class_1_rate', 0):.3f} → "
                f"Current: {od.get('current_class_1_rate', 0):.3f}"
            )

# ── Investigations ──────────────────────────────────────────────────────────
elif page == "Investigations":
    st.title("Investigations")
    investigations = _get(f"{AGENT_URL}/investigations") or []
    for inv in investigations:
        status = inv.get("status", "unknown")
        icon = {"open": "🔵", "awaiting_approval": "🟡", "resolved": "✅"}.get(status, "⚪")
        with st.expander(f"{icon} {inv['id'][:8]}… — {status}"):
            st.markdown(inv.get("summary_md") or "_No summary yet._")
            if inv.get("updated_at"):
                st.caption(f"Updated: {inv['updated_at']}")

# ── Queue ───────────────────────────────────────────────────────────────────
elif page == "Queue":
    st.title("Queue Status")
    st.info("Queue metrics require Redis info API — connect via arq dashboard or extend this page.")
    st.subheader("Dead-Letter Queue (DLQ)")
    # DLQ items would be read from Redis drift_actions:dlq — placeholder
    st.write("DLQ inspection: run `redis-cli lrange drift_actions:dlq 0 -1`")

# ── HIL Inbox ───────────────────────────────────────────────────────────────
elif page == "HIL Inbox":
    st.title("Human-in-the-Loop Inbox")
    pending = _get(f"{AGENT_URL}/hil/approvals") or []

    if not pending:
        st.success("No pending approvals.")
    else:
        for approval in pending:
            investigation_id = approval["investigation_id"]
            approval_id = approval["id"]
            st.warning(f"**Approval required** — Investigation `{investigation_id[:8]}…`")
            st.markdown(approval.get("summary_md") or approval.get("rationale") or "")
            col_a, col_r = st.columns(2)
            with col_a:
                if st.button("✅ Approve", key=f"approve_{approval_id}"):
                    _post(
                        f"{AGENT_URL}/hil/approve",
                        {
                            "investigation_id": investigation_id,
                            "hil_approval_id": approval_id,
                            "decision": "approved",
                        },
                    )
                    st.success("Approved — agent resuming.")
            with col_r:
                if st.button("❌ Reject", key=f"reject_{approval_id}"):
                    _post(
                        f"{AGENT_URL}/hil/approve",
                        {
                            "investigation_id": investigation_id,
                            "hil_approval_id": approval_id,
                            "decision": "rejected",
                        },
                    )
                    st.info("Rejected.")

# ── Registry ─────────────────────────────────────────────────────────────────
elif page == "Registry":
    st.title("Model Registry")
    st.info("MLflow registry — open http://mlflow:5000 or embed via iframe for full view.")
    st.write("Key metrics visible after `make train` registers a model.")
