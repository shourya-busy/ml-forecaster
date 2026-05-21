"""custom_run_configs table for the Custom Run panel

Revision ID: 0004_custom_run_configs
Revises: 0003_celery_task_id
Create Date: 2026-05-21
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0004_custom_run_configs"
down_revision: Union[str, None] = "0003_celery_task_id"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _json_type():
    return sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "custom_run_configs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(128), nullable=False, unique=True),
        sa.Column("instance", sa.String(256), nullable=False),
        sa.Column("metric", sa.String(64), nullable=False),
        sa.Column("horizon", sa.String(32), nullable=False),
        sa.Column("algorithms", _json_type()),
        sa.Column("anomaly_filter", _json_type()),
        sa.Column("note", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  nullable=False, server_default=sa.func.now()),
        sa.Column("last_used_at", sa.DateTime(timezone=True)),
        sa.Column("run_count", sa.Integer, nullable=False, server_default="0"),
    )
    op.create_index("ix_custom_run_configs_name", "custom_run_configs", ["name"])


def downgrade() -> None:
    op.drop_index("ix_custom_run_configs_name", table_name="custom_run_configs")
    op.drop_table("custom_run_configs")
