"""add data_stats JSON column to training_runs

Captures the training-data footprint per run (fetched points, used
points after anomaly filter, time range covered).

Revision ID: 0005_data_stats
Revises: 0004_custom_run_configs
Create Date: 2026-05-21
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0005_data_stats"
down_revision: Union[str, None] = "0004_custom_run_configs"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _json_type():
    return sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.add_column("training_runs", sa.Column("data_stats", _json_type(), nullable=True))


def downgrade() -> None:
    op.drop_column("training_runs", "data_stats")
