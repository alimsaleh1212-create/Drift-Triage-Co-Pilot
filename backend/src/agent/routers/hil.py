"""HIL approval endpoints: POST /hil/approve, GET /hil/approvals."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from agent.deps.db import get_session
from agent.deps.graph import get_graph, graph_resume_config
from core.logging import get_logger

router = APIRouter()
log = get_logger(__name__)


class HILApprovalRequest(BaseModel):
    investigation_id: str
    hil_approval_id: str
    decision: str  # "approved" | "rejected"


class HILApprovalResponse(BaseModel):
    investigation_id: str
    decision: str
    status: str


@router.post("/hil/approve", response_model=HILApprovalResponse)
async def approve_hil(
    payload: HILApprovalRequest,
    graph=Depends(get_graph),
) -> HILApprovalResponse:
    """Record HIL decision and resume the paused investigation graph."""
    if payload.decision not in ("approved", "rejected"):
        raise HTTPException(
            status_code=400, detail="decision must be 'approved' or 'rejected'"
        )

    log.info(
        "hil.decision",
        investigation_id=payload.investigation_id,
        decision=payload.decision,
    )

    async with get_session() as session:
        from sqlalchemy import text

        await session.execute(
            text(
                "UPDATE hil_approvals "
                "SET status = :status, decision = :decision, decided_at = now() "
                "WHERE id = :approval_id AND investigation_id = :investigation_id"
            ),
            {
                "status": payload.decision,
                "decision": payload.decision,
                "approval_id": payload.hil_approval_id,
                "investigation_id": payload.investigation_id,
            },
        )
        if payload.decision == "rejected":
            await session.execute(
                text(
                    "UPDATE investigations "
                    "SET status = 'resolved', updated_at = now() "
                    "WHERE id = :investigation_id"
                ),
                {"investigation_id": payload.investigation_id},
            )
        await session.commit()

    if payload.decision == "approved":
        config = graph_resume_config(investigation_id=payload.investigation_id)
        await graph.ainvoke(None, config=config)

    return HILApprovalResponse(
        investigation_id=payload.investigation_id,
        decision=payload.decision,
        status="resumed" if payload.decision == "approved" else "rejected",
    )


@router.get("/hil/approvals")
async def list_hil_approvals() -> list[dict[str, Any]]:
    """List pending HIL approvals with their real approval IDs."""
    from sqlalchemy import text

    async with get_session() as session:
        result = await session.execute(
            text(
                "SELECT "
                "h.id, h.investigation_id, h.action, h.rationale, h.model_version, "
                "h.status, h.created_at, i.summary_md, i.updated_at "
                "FROM hil_approvals h "
                "JOIN investigations i ON i.id = h.investigation_id "
                "WHERE h.status = 'pending' "
                "ORDER BY h.created_at DESC "
                "LIMIT 50"
            )
        )
        return [dict(r._mapping) for r in result]
