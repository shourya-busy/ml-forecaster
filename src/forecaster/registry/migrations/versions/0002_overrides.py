"""settings + target overrides for UI-managed configuration

Revision ID: 0002_overrides
Revises: 0001_initial
Create Date: 2026-05-21
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0002_overrides"
down_revision: Union[str, None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _json_type():
    return sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "settings_overrides",
        sa.Column("key", sa.String(256), primary_key=True),
        sa.Column("value", _json_type(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_by", sa.String(128)),
    )

    op.create_table(
        "target_overrides",
        sa.Column("instance", sa.String(256), primary_key=True),
        sa.Column("metric", sa.String(64), primary_key=True),
        sa.Column("horizon", sa.String(32), primary_key=True),
        sa.Column("enabled", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("schedule_cron", sa.String(64)),
        sa.Column("note", sa.Text()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_by", sa.String(128)),
    )
    op.create_index("ix_target_overrides_enabled", "target_overrides", ["enabled"])


def downgrade() -> None:
    op.drop_index("ix_target_overrides_enabled", table_name="target_overrides")
    op.drop_table("target_overrides")
    op.drop_table("settings_overrides")
