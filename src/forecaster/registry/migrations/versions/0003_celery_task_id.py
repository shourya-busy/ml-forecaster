"""add celery_task_id to training_runs

Revision ID: 0003_celery_task_id
Revises: 0002_overrides
Create Date: 2026-05-21
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0003_celery_task_id"
down_revision: Union[str, None] = "0002_overrides"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "training_runs",
        sa.Column("celery_task_id", sa.String(128), nullable=True),
    )
    op.create_index(
        "ix_training_runs_celery_task_id",
        "training_runs",
        ["celery_task_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_training_runs_celery_task_id", table_name="training_runs")
    op.drop_column("training_runs", "celery_task_id")
