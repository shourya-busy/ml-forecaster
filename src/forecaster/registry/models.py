"""SQLAlchemy models for the registry.

A TrainingRun groups everything produced for one (instance, metric,
horizon) at a moment in time: per-algo backtest scores, per-algo
forecasts, a final ranking row marking the winner, and a model artifact
row per algo pointing at the file on disk.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# JSONB on Postgres, JSON on every other dialect (SQLite, MySQL...).
JSON_FIELD = JSON().with_variant(JSONB(), "postgresql")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    instance: Mapped[str] = mapped_column(String(256), index=True)
    metric: Mapped[str] = mapped_column(String(64), index=True)
    horizon: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    config_snapshot: Mapped[dict] = mapped_column(JSON_FIELD, default=dict)

    metrics: Mapped[list["RunMetric"]] = relationship(back_populates="run", cascade="all, delete-orphan")
    artifacts: Mapped[list["ModelArtifact"]] = relationship(back_populates="run", cascade="all, delete-orphan")
    forecasts: Mapped[list["Forecast"]] = relationship(back_populates="run", cascade="all, delete-orphan")
    ranking: Mapped["Ranking | None"] = relationship(back_populates="run", cascade="all, delete-orphan", uselist=False)

    __table_args__ = (
        Index("ix_runs_target_time", "instance", "metric", "horizon", "started_at"),
    )


class RunMetric(Base):
    """Per-algo backtest scores for a given run."""

    __tablename__ = "run_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("training_runs.id", ondelete="CASCADE"), index=True)
    algo: Mapped[str] = mapped_column(String(64), index=True)
    score_metric: Mapped[str] = mapped_column(String(16))  # mae|rmse|mape|smape|r2
    value: Mapped[float] = mapped_column(Float)
    fold: Mapped[int] = mapped_column(Integer, default=-1)  # -1 = averaged

    run: Mapped[TrainingRun] = relationship(back_populates="metrics")

    __table_args__ = (
        UniqueConstraint("run_id", "algo", "score_metric", "fold", name="uq_run_metric"),
    )


class ModelArtifact(Base):
    __tablename__ = "model_artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("training_runs.id", ondelete="CASCADE"), index=True)
    algo: Mapped[str] = mapped_column(String(64), index=True)
    path: Mapped[str] = mapped_column(String(512))
    size_bytes: Mapped[int] = mapped_column(Integer)
    train_duration_seconds: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    run: Mapped[TrainingRun] = relationship(back_populates="artifacts")


class Forecast(Base):
    """Stored forecast points; one row per future timestamp per algo."""

    __tablename__ = "forecasts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("training_runs.id", ondelete="CASCADE"), index=True)
    instance: Mapped[str] = mapped_column(String(256), index=True)
    metric: Mapped[str] = mapped_column(String(64), index=True)
    horizon: Mapped[str] = mapped_column(String(32), index=True)
    algo: Mapped[str] = mapped_column(String(64), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    point: Mapped[float] = mapped_column(Float)
    lower: Mapped[float | None] = mapped_column(Float, nullable=True)
    upper: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_best: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    run: Mapped[TrainingRun] = relationship(back_populates="forecasts")

    __table_args__ = (
        Index(
            "ix_forecast_lookup",
            "instance", "metric", "horizon", "ts",
        ),
        Index("ix_forecast_best_lookup", "instance", "metric", "horizon", "is_best", "ts"),
    )


class Ranking(Base):
    """Final ranking row per run — the winner and the composite vector."""

    __tablename__ = "rankings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("training_runs.id", ondelete="CASCADE"), unique=True)
    instance: Mapped[str] = mapped_column(String(256), index=True)
    metric: Mapped[str] = mapped_column(String(64), index=True)
    horizon: Mapped[str] = mapped_column(String(32), index=True)
    winning_algo: Mapped[str] = mapped_column(String(64), index=True)
    ranked: Mapped[list[dict]] = mapped_column(JSON_FIELD)
    # full ranking: [{algo, rank, composite, raw_scores, normalised_scores}, ...]

    run: Mapped[TrainingRun] = relationship(back_populates="ranking")
