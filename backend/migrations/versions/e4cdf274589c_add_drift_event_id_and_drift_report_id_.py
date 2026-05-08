"""add drift_event_id and drift_report_id to investigations

Revision ID: e4cdf274589c
Revises: 2d9b7f4e6a31
Create Date: 2026-05-08 21:20:56.937567

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e4cdf274589c"
down_revision: Union[str, None] = "2d9b7f4e6a31"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "investigations",
        sa.Column("drift_event_id", sa.Text(), nullable=True),
    )
    op.add_column(
        "investigations",
        sa.Column("drift_report_id", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("investigations", "drift_report_id")
    op.drop_column("investigations", "drift_event_id")
