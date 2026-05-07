"""SQLAlchemy ORM models — schema source of truth for Alembic migrations."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    DateTime,
    Double,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    """Rolling-window prediction log used for drift detection."""

    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    features: Mapped[dict] = mapped_column(JSONB, nullable=False)
    label: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    probability: Mapped[float] = mapped_column(Double, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index(
            "predictions_created_at_idx",
            "created_at",
            postgresql_ops={"created_at": "DESC"},
        ),
    )


class Investigation(Base):
    """Agent investigation state (LangGraph checkpoints live separately)."""

    __tablename__ = "investigations"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    drift_event_id: Mapped[str | None] = mapped_column(Text, nullable=True, unique=True)
    drift_report_id: Mapped[str | None] = mapped_column(
        Text, nullable=True, unique=True
    )
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="open")
    summary_md: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class HILApproval(Base):
    """Human-in-the-loop approval requests."""

    __tablename__ = "hil_approvals"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    investigation_id: Mapped[str] = mapped_column(
        Text, ForeignKey("investigations.id"), nullable=False
    )
    action: Mapped[str] = mapped_column(Text, nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    model_version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    decision: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    decided_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class PromotionEvent(Base):
    """Audit log for model promotions."""

    __tablename__ = "promotion_events"

    id: Mapped[str] = mapped_column(
        Text, primary_key=True, server_default=func.gen_random_uuid()
    )
    model_name: Mapped[str] = mapped_column(Text, nullable=False)
    promoted_version: Mapped[int] = mapped_column(Integer, nullable=False)
    previous_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    investigation_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class DriftAlertState(Base):
    """Last successfully emitted drift webhook state for restart-safe delivery."""

    __tablename__ = "drift_alert_state"

    key: Mapped[str] = mapped_column(Text, primary_key=True)
    last_severity: Mapped[str] = mapped_column(Text, nullable=False)
    last_report_id: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
