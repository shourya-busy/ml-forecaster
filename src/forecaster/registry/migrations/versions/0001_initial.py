"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-05-19
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "training_runs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("instance", sa.String(256), nullable=False),
        sa.Column("metric", sa.String(64), nullable=False),
        sa.Column("horizon", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("duration_seconds", sa.Float()),
        sa.Column("error", sa.Text()),
        sa.Column("config_snapshot", postgresql.JSONB, nullable=False, server_default="{}"),
    )
    op.create_index("ix_training_runs_instance", "training_runs", ["instance"])
    op.create_index("ix_training_runs_metric", "training_runs", ["metric"])
    op.create_index("ix_training_runs_horizon", "training_runs", ["horizon"])
    op.create_index("ix_training_runs_status", "training_runs", ["status"])
    op.create_index("ix_runs_target_time", "training_runs", ["instance", "metric", "horizon", "started_at"])

    op.create_table(
        "run_metrics",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.Integer, sa.ForeignKey("training_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("algo", sa.String(64), nullable=False),
        sa.Column("score_metric", sa.String(16), nullable=False),
        sa.Column("value", sa.Float, nullable=False),
        sa.Column("fold", sa.Integer, nullable=False, server_default="-1"),
        sa.UniqueConstraint("run_id", "algo", "score_metric", "fold", name="uq_run_metric"),
    )
    op.create_index("ix_run_metrics_run_id", "run_metrics", ["run_id"])
    op.create_index("ix_run_metrics_algo", "run_metrics", ["algo"])

    op.create_table(
        "model_artifacts",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.Integer, sa.ForeignKey("training_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("algo", sa.String(64), nullable=False),
        sa.Column("path", sa.String(512), nullable=False),
        sa.Column("size_bytes", sa.Integer, nullable=False),
        sa.Column("train_duration_seconds", sa.Float, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_model_artifacts_run_id", "model_artifacts", ["run_id"])
    op.create_index("ix_model_artifacts_algo", "model_artifacts", ["algo"])

    op.create_table(
        "forecasts",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.Integer, sa.ForeignKey("training_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("instance", sa.String(256), nullable=False),
        sa.Column("metric", sa.String(64), nullable=False),
        sa.Column("horizon", sa.String(32), nullable=False),
        sa.Column("algo", sa.String(64), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("point", sa.Float, nullable=False),
        sa.Column("lower", sa.Float),
        sa.Column("upper", sa.Float),
        sa.Column("is_best", sa.Boolean, nullable=False, server_default=sa.false()),
    )
    op.create_index("ix_forecasts_run_id", "forecasts", ["run_id"])
    op.create_index("ix_forecasts_instance", "forecasts", ["instance"])
    op.create_index("ix_forecasts_metric", "forecasts", ["metric"])
    op.create_index("ix_forecasts_horizon", "forecasts", ["horizon"])
    op.create_index("ix_forecasts_algo", "forecasts", ["algo"])
    op.create_index("ix_forecasts_ts", "forecasts", ["ts"])
    op.create_index("ix_forecasts_is_best", "forecasts", ["is_best"])
    op.create_index("ix_forecast_lookup", "forecasts", ["instance", "metric", "horizon", "ts"])
    op.create_index("ix_forecast_best_lookup", "forecasts", ["instance", "metric", "horizon", "is_best", "ts"])

    op.create_table(
        "rankings",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.Integer, sa.ForeignKey("training_runs.id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("instance", sa.String(256), nullable=False),
        sa.Column("metric", sa.String(64), nullable=False),
        sa.Column("horizon", sa.String(32), nullable=False),
        sa.Column("winning_algo", sa.String(64), nullable=False),
        sa.Column("ranked", postgresql.JSONB, nullable=False),
    )
    op.create_index("ix_rankings_instance", "rankings", ["instance"])
    op.create_index("ix_rankings_metric", "rankings", ["metric"])
    op.create_index("ix_rankings_horizon", "rankings", ["horizon"])
    op.create_index("ix_rankings_winning_algo", "rankings", ["winning_algo"])


def downgrade() -> None:
    op.drop_table("rankings")
    op.drop_table("forecasts")
    op.drop_table("model_artifacts")
    op.drop_table("run_metrics")
    op.drop_table("training_runs")
