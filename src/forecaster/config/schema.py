"""Pydantic schemas for all configuration.

A single Settings object aggregates default.yaml, data_sources.yaml,
targets.yaml and exposition.yaml. Validated at startup; anything missing
or mis-typed will raise loudly instead of failing deep inside a worker.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------- horizons ----------

class HorizonSpec(BaseModel):
    step: str = Field(..., description="Pandas-compatible step, e.g. '1min', '5min', '1h'")
    horizon: str = Field(..., description="Total prediction horizon, e.g. '1h', '24h', '7d'")
    retrain: str = Field(..., description="Cron expression in 5-field form")
    lookback_days: int | None = None


# ---------- training ----------

class Parallelism(BaseModel):
    workers: int = 4
    algos_per_job: int = 4
    fetch_jitter_seconds: int = 30


class AnomalyFilter(BaseModel):
    """Optional preprocessing step: drop outliers from the training series
    before fitting models.

    Use this when your metric has occasional deployment spikes, monitoring
    blips, or other artifacts that would otherwise contaminate the fit.
    The default is OFF — never silently mutate user data.
    """

    enabled: bool = False
    method: Literal["isolation_forest"] = "isolation_forest"
    contamination: float = Field(
        0.02,
        ge=0.001,
        le=0.5,
        description="Expected fraction of outliers. 0.02 = drop the most-anomalous 2%.",
    )
    window: int = Field(
        1,
        ge=1,
        description=(
            "Lag-window size used as the feature vector for the detector. "
            "Default 1 (point-level outliers — usually the right choice for "
            "monitoring metrics). Increase to 12-48 if your outliers are "
            "*shapes* (brief load patterns) rather than single spikes."
        ),
    )


class TrainingConfig(BaseModel):
    lookback_days: int = 30
    backtest_folds: int = 5
    backtest_holdout_fraction: float = 0.1
    per_algo_lookback_override: dict[str, int] = Field(default_factory=dict)
    parallelism: Parallelism = Field(default_factory=Parallelism)
    confidence_alpha: float = 0.05  # for 95% intervals
    max_artifact_versions_kept: int = 3
    anomaly_filter: AnomalyFilter = Field(default_factory=AnomalyFilter)


# ---------- algorithms ----------

class AlgorithmConfig(BaseModel):
    enabled: list[str] = Field(default_factory=list)
    defaults: dict[str, dict[str, Any]] = Field(default_factory=dict)
    per_metric: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Optional per-metric algorithm shortlist. Each list must be a "
            "subset of `enabled` and contain only registered algorithms. "
            "Metrics absent here fall back to the full `enabled` list."
        ),
    )

    @field_validator("enabled")
    @classmethod
    def _unique(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("algorithms.enabled contains duplicates")
        return v

    @model_validator(mode="after")
    def _validate_per_metric(self) -> "AlgorithmConfig":
        # Defer the registry import to here so a circular dependency at
        # import time (config -> models -> ...) doesn't bite us.
        from ..models.registry import REGISTRY

        enabled = set(self.enabled)
        for metric, shortlist in self.per_metric.items():
            if not shortlist:
                raise ValueError(
                    f"algorithms.per_metric.{metric} is empty; "
                    "remove the key to fall back to algorithms.enabled"
                )
            if len(set(shortlist)) != len(shortlist):
                raise ValueError(f"algorithms.per_metric.{metric} contains duplicates")
            unknown = [a for a in shortlist if a not in REGISTRY]
            if unknown:
                raise ValueError(
                    f"algorithms.per_metric.{metric} references unregistered "
                    f"algorithm(s): {unknown}"
                )
            not_in_enabled = [a for a in shortlist if a not in enabled]
            if not_in_enabled:
                raise ValueError(
                    f"algorithms.per_metric.{metric}: {not_in_enabled} "
                    "must also appear in algorithms.enabled"
                )
        return self


# ---------- ranking ----------

RankDirection = Literal["min", "max"]


class RankingConfig(BaseModel):
    metrics: list[str]
    weights: dict[str, float]
    direction: dict[str, RankDirection]

    @field_validator("weights")
    @classmethod
    def _weights_sum_close_to_one(cls, v: dict[str, float]) -> dict[str, float]:
        s = sum(v.values())
        if not 0.99 <= s <= 1.01:
            raise ValueError(f"ranking.weights must sum to ~1.0, got {s}")
        return v


# ---------- targets / metrics_to_forecast ----------

class TargetsConfig(BaseModel):
    discovery: Literal["static", "promql"] = "static"
    static_instances: list[str] = Field(default_factory=list)
    discovery_query: str | None = None  # e.g. 'group by (instance) (up{job="netdata"})'
    instance_label: str = "instance"


class MetricsToForecast(BaseModel):
    """Map of friendly name -> PromQL expression.

    The PromQL must return a per-instance series; the instance label is
    used as the per-server identifier.
    """

    queries: dict[str, str]


# ---------- data sources ----------

class DataSourceEndpoint(BaseModel):
    kind: Literal["prometheus", "mimir"]
    base_url: str
    timeout_seconds: int = 30
    tenant_id: str | None = None  # Mimir multi-tenant header
    bearer_token: str | None = None
    basic_auth_user: str | None = None
    basic_auth_password: str | None = None
    verify_tls: bool = True


class DataSourcesConfig(BaseModel):
    active: str  # name of the endpoint to use
    endpoints: dict[str, DataSourceEndpoint]

    @field_validator("endpoints")
    @classmethod
    def _at_least_one(cls, v: dict[str, DataSourceEndpoint]) -> dict[str, DataSourceEndpoint]:
        if not v:
            raise ValueError("data_sources.endpoints must contain at least one entry")
        return v


# ---------- exposition (cardinality control) ----------

class ExpositionEmit(BaseModel):
    best_model_forecast: bool = True
    best_model_bounds: bool = True
    per_model_forecast: bool = False
    per_model_bounds: bool = False
    ranking_scores: bool = True
    training_run_timestamps: bool = True
    training_durations: bool = True
    diagnostics: bool = True  # emit forecaster_winner + forecaster_winner_unique_recent


class ExpositionLabels(BaseModel):
    include_model_version: bool = False
    include_horizon: bool = True


class DiagnosticsConfig(BaseModel):
    recent_window_runs: int = Field(
        10, description="K for forecaster_winner_unique_recent — looks back this many runs"
    )


class ExpositionConfig(BaseModel):
    emit: ExpositionEmit = Field(default_factory=ExpositionEmit)
    labels: ExpositionLabels = Field(default_factory=ExpositionLabels)
    series_per_forecast: int = 60  # how many future points to emit per series
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)


# ---------- artifact store ----------

class ArtifactStoreConfig(BaseModel):
    kind: Literal["volume", "s3"] = "volume"
    volume_path: str = "/var/lib/forecaster/models"
    s3_bucket: str | None = None
    s3_prefix: str = ""


# ---------- top level ----------

class Settings(BaseModel):
    horizons: dict[str, HorizonSpec]
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    algorithms: AlgorithmConfig
    ranking: RankingConfig
    targets: TargetsConfig
    metrics_to_forecast: MetricsToForecast
    data_sources: DataSourcesConfig
    exposition: ExpositionConfig = Field(default_factory=ExpositionConfig)
    artifact_store: ArtifactStoreConfig = Field(default_factory=ArtifactStoreConfig)

    # runtime / infra
    database_url: str = "postgresql+psycopg://forecaster:forecaster@postgres:5432/forecaster"
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/1"
    log_level: str = "INFO"
    use_cuda: bool = False

    # All UI timestamps render in this timezone. Cron expressions in
    # `horizons.*.retrain` are also interpreted in this zone. Storage and
    # JSON API responses stay in UTC (ISO 8601 with offset).
    display_timezone: str = "Asia/Kolkata"

    @field_validator("horizons")
    @classmethod
    def _horizons_not_empty(cls, v: dict[str, HorizonSpec]) -> dict[str, HorizonSpec]:
        if not v:
            raise ValueError("at least one horizon must be configured")
        return v
