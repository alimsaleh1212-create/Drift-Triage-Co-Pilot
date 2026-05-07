"""persist drift webhook state and investigation event ids

Revision ID: 2d9b7f4e6a31
Revises: b8b18f902dcf
Create Date: 2026-05-08 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "2d9b7f4e6a31"
down_revision: Union[str, None] = "b8b18f902dcf"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "drift_alert_state",
        sa.Column("key", sa.Text(), nullable=False),
        sa.Column("last_severity", sa.Text(), nullable=False),
        sa.Column("last_report_id", sa.Text(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("key"),
    )
    op.add_column(
        "investigations",
        sa.Column("drift_event_id", sa.Text(), nullable=True),
    )
    op.add_column(
        "investigations",
        sa.Column("drift_report_id", sa.Text(), nullable=True),
    )
    op.create_unique_constraint(
        "investigations_drift_event_id_key",
        "investigations",
        ["drift_event_id"],
    )
    op.create_unique_constraint(
        "investigations_drift_report_id_key",
        "investigations",
        ["drift_report_id"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(
        "investigations_drift_report_id_key",
        "investigations",
        type_="unique",
    )
    op.drop_constraint(
        "investigations_drift_event_id_key",
        "investigations",
        type_="unique",
    )
    op.drop_column("investigations", "drift_report_id")
    op.drop_column("investigations", "drift_event_id")
    op.drop_table("drift_alert_state")
