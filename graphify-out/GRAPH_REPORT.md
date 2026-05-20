# Graph Report - .  (2026-05-20)

## Corpus Check
- Corpus is ~17,835 words - fits in a single context window. You may not need a graph.

## Summary
- 460 nodes · 680 edges · 48 communities (28 shown, 20 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 56 edges (avg confidence: 0.69)
- Token cost: 57,142 input · 2,279 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Config Loading|Config Loading]]
- [[_COMMUNITY_Tabular Features & Forecast Result|Tabular Features & Forecast Result]]
- [[_COMMUNITY_Registry Models|Registry Models]]
- [[_COMMUNITY_Data Source Interface|Data Source Interface]]
- [[_COMMUNITY_Backtesting & Walk-Forward CV|Backtesting & Walk-Forward CV]]
- [[_COMMUNITY_Artifact Storage|Artifact Storage]]
- [[_COMMUNITY_Grafana Dashboard JSON|Grafana Dashboard JSON]]
- [[_COMMUNITY_Pydantic Config Schema|Pydantic Config Schema]]
- [[_COMMUNITY_Diagnostics Endpoints|Diagnostics Endpoints]]
- [[_COMMUNITY_FastAPI App & Prometheus Export|FastAPI App & Prometheus Export]]
- [[_COMMUNITY_Per-Metric Shortlist Tests|Per-Metric Shortlist Tests]]
- [[_COMMUNITY_Model Ranking|Model Ranking]]
- [[_COMMUNITY_LSTM Forecaster|LSTM Forecaster]]
- [[_COMMUNITY_Docker Compose Services|Docker Compose Services]]
- [[_COMMUNITY_LightGBM Forecaster|LightGBM Forecaster]]
- [[_COMMUNITY_Prophet Forecaster|Prophet Forecaster]]
- [[_COMMUNITY_Forecaster Protocol|Forecaster Protocol]]
- [[_COMMUNITY_Recursive Lag Forecasting Tests|Recursive Lag Forecasting Tests]]
- [[_COMMUNITY_Models Smoke Test|Models Smoke Test]]
- [[_COMMUNITY_ARIMA Forecaster|ARIMA Forecaster]]
- [[_COMMUNITY_Base Forecaster|Base Forecaster]]
- [[_COMMUNITY_ETS Forecaster|ETS Forecaster]]
- [[_COMMUNITY_Holt-Winters Forecaster|Holt-Winters Forecaster]]
- [[_COMMUNITY_Naive Forecaster|Naive Forecaster]]
- [[_COMMUNITY_Seasonal Naive Forecaster|Seasonal Naive Forecaster]]
- [[_COMMUNITY_XGBoost Forecaster|XGBoost Forecaster]]
- [[_COMMUNITY_Pipeline E2E Test|Pipeline E2E Test]]
- [[_COMMUNITY_Synthetic Series Fixture|Synthetic Series Fixture]]
- [[_COMMUNITY_Alembic Migration Env|Alembic Migration Env]]
- [[_COMMUNITY_Fallback Synthetic Series|Fallback Synthetic Series]]
- [[_COMMUNITY_Forecaster Package Init|Forecaster Package Init]]
- [[_COMMUNITY_Test Conftest Fixtures|Test Conftest Fixtures]]
- [[_COMMUNITY_Protocol fit()|Protocol: fit()]]
- [[_COMMUNITY_Protocol predict()|Protocol: predict()]]
- [[_COMMUNITY_Protocol predict_interval()|Protocol: predict_interval()]]
- [[_COMMUNITY_Protocol save()|Protocol: save()]]
- [[_COMMUNITY_Protocol load()|Protocol: load()]]
- [[_COMMUNITY_Protocol delete()|Protocol: delete()]]
- [[_COMMUNITY_TSDataSource.fetch_range()|TSDataSource.fetch_range()]]
- [[_COMMUNITY_TSDataSource.discover_instances()|TSDataSource.discover_instances()]]
- [[_COMMUNITY_Architecture Overview Doc|Architecture Overview Doc]]
- [[_COMMUNITY_CLAUDE.md Project Guidance|CLAUDE.md Project Guidance]]

## God Nodes (most connected - your core abstractions)
1. `RegistryRepo` - 29 edges
2. `BaseForecaster` - 15 edges
3. `run_pipeline()` - 14 edges
4. `session()` - 14 edges
5. `get_settings()` - 13 edges
6. `PrometheusClient` - 11 edges
7. `load_settings()` - 10 edges
8. `ProphetForecaster` - 9 edges
9. `AlgorithmConfig` - 8 edges
10. `LSTMForecaster` - 8 edges

## Surprising Connections (you probably didn't know these)
- `test_all_metrics_returns_5_keys()` --calls--> `all_metrics()`  [INFERRED]
  tests/unit/test_metrics.py → src/forecaster/evaluation/metrics.py
- `test_nan_safe()` --calls--> `all_metrics()`  [INFERRED]
  tests/unit/test_metrics.py → src/forecaster/evaluation/metrics.py
- `test_build_lag_frame_shape()` --calls--> `build_lag_frame()`  [INFERRED]
  tests/unit/test_lag_features.py → src/forecaster/features/lag_features.py
- `make_cfg()` --calls--> `RankingConfig`  [INFERRED]
  tests/unit/test_ranking.py → src/forecaster/config/schema.py
- `test_default_config_loads()` --calls--> `load_settings()`  [INFERRED]
  tests/unit/test_config.py → src/forecaster/config/loader.py

## Communities (48 total, 20 thin omitted)

### Community 0 - "Config Loading"
Cohesion: 0.06
Nodes (36): settings_dep(), _apply_env_overrides(), _coerce(), _deep_merge(), get_settings(), load_settings(), YAML config loader.  Loads four YAML files from FORECASTER_CONFIG_DIR (default:, Force-reload and replace the cached settings (used on SIGHUP). (+28 more)

### Community 1 - "Tabular Features & Forecast Result"
Cohesion: 0.07
Nodes (18): Shared lag / calendar feature builder for tabular ML models.  Models like XGBoos, ARIMA via statsmodels.  Uses a small grid search over (p,d,q) when pmdarima isn', ForecastResult, Forecaster protocol.  Every algorithm in src/forecaster/models/ implements the s, Output of a single (algo, instance, metric, horizon) run., ETS / Exponential smoothing (statsmodels)., Holt-Winters triple exponential smoothing., All built-in forecasters auto-register here. (+10 more)

### Community 2 - "Registry Models"
Cohesion: 0.09
Nodes (20): DeclarativeBase, Base, Forecast, ModelArtifact, Ranking, SQLAlchemy models for the registry.  A TrainingRun groups everything produced fo, Final ranking row per run — the winner and the composite vector., Per-algo backtest scores for a given run. (+12 more)

### Community 3 - "Data Source Interface"
Cohesion: 0.09
Nodes (22): FetchError, Pluggable time-series data-source interface.  A TSDataSource fetches a single se, Raised when a data-source fetch fails for non-transient reasons., Per-instance series result.      The DataFrame has a tz-aware DatetimeIndex (UTC, ABC for any time-series fetcher., TimeSeries, TSDataSource, Build a TSDataSource from configuration. (+14 more)

### Community 4 - "Backtesting & Walk-Forward CV"
Cohesion: 0.10
Nodes (27): BacktestResult, _fold_indices(), Walk-forward cross-validation.  We split a 1-D series into K expanding-window fo, Return list of (train_end_inclusive, test_end_exclusive) indices., Run walk-forward CV; return (averaged_scores, per_fold_scores)., walk_forward(), all_metrics(), _clean() (+19 more)

### Community 5 - "Artifact Storage"
Cohesion: 0.08
Nodes (15): make_data_source(), ArtifactStore, Artifact storage abstraction.  Default impl writes pickled artifacts to a local, VolumeArtifactStore, Training-run endpoints.  POST /runs              — kick off a training run (asyn, Run the pipeline in-process. Useful in dev / for tests., RunRequest, trigger_sync() (+7 more)

### Community 6 - "Grafana Dashboard JSON"
Cohesion: 0.08
Nodes (24): annotations, list, description, editable, fiscalYearStartMonth, graphTooltip, id, links (+16 more)

### Community 7 - "Pydantic Config Schema"
Cohesion: 0.14
Nodes (17): BaseModel, ArtifactStoreConfig, DataSourceEndpoint, DataSourcesConfig, DiagnosticsConfig, ExpositionConfig, ExpositionEmit, ExpositionLabels (+9 more)

### Community 8 - "Diagnostics Endpoints"
Cohesion: 0.11
Nodes (6): FastAPI dependency providers., _repo(), repo_dep(), Diagnostics endpoints — tools for trusting / debugging the auto-ranking.  Three, Forecast endpoints — full JSON payload with bands., Algorithm-registry introspection.

### Community 9 - "FastAPI App & Prometheus Export"
Cohesion: 0.12
Nodes (14): create_app(), FastAPI app factory + uvicorn entrypoint., _build_registry(), metrics(), Prometheus exposition endpoint.  Emits gauges so Prometheus can scrape and Grafa, Build a fresh registry per scrape.      We deliberately rebuild on every /metric, Prometheus Metrics Router, Hit /metrics on a populated SQLite DB; confirm Prom-format output. (+6 more)

### Community 10 - "Per-Metric Shortlist Tests"
Cohesion: 0.33
Nodes (11): AlgorithmConfig, _base_kwargs(), Validation tests for algorithms.per_metric shortlists., The defaults shipped in config/default.yaml must be valid., test_per_metric_duplicate_rejected(), test_per_metric_empty_list_rejected(), test_per_metric_missing_metric_falls_back(), test_per_metric_non_subset_rejected() (+3 more)

### Community 11 - "Model Ranking"
Cohesion: 0.31
Nodes (9): _normalise(), rank_models(), RankedModel, Weighted composite ranking of candidate models.  Each ranking metric is min-max, Rank candidates from a {algo: {metric: score}} mapping., make_cfg(), test_handles_single_candidate(), test_perfect_winner_first() (+1 more)

### Community 12 - "LSTM Forecaster"
Cohesion: 0.33
Nodes (3): _device(), LSTMForecaster, Small LSTM forecaster (PyTorch).  Trains on normalized lag sequences and forecas

### Community 13 - "Docker Compose Services"
Cohesion: 0.29
Nodes (7): Worker Service (GPU), API Service, Migration Service, Postgres Service, Redis Service, Scheduler Service, Worker Service

### Community 14 - "LightGBM Forecaster"
Cohesion: 0.33
Nodes (3): build_lag_frame(), Build X, y matrices with `lags` autoregressive lag features + calendar., LightGBMForecaster

### Community 16 - "Forecaster Protocol"
Cohesion: 0.33
Nodes (3): Forecaster, Protocol every algorithm must follow.      Implementations must be picklable so, Minimum number of points needed; default 1.

### Community 17 - "Recursive Lag Forecasting Tests"
Cohesion: 0.47
Nodes (5): Recursive one-step-ahead forecasting.      `predict_one` takes the feature vecto, recursive_forecast(), _series(), test_build_lag_frame_shape(), test_recursive_forecast_returns_correct_length()

### Community 18 - "Models Smoke Test"
Cohesion: 0.47
Nodes (5): build(), _dep_missing(), Smoke test: every registered model can fit + predict on synthetic data.  We use, test_model_fit_predict_smoke(), test_predict_interval_returns_bounds()

### Community 20 - "Base Forecaster"
Cohesion: 0.40
Nodes (3): BaseForecaster, predict(), Convenience base with simple residual-based prediction intervals.      Subclasse

### Community 23 - "Naive Forecaster"
Cohesion: 0.40
Nodes (3): BaseForecaster, NaiveForecaster, Predict the last observed value for all future steps.

### Community 26 - "Pipeline E2E Test"
Cohesion: 0.40
Nodes (3): End-to-end pipeline test with SQLite + monkey-patched data source.  Exercises: f, Run pipeline against a stubbed data source., test_pipeline_end_to_end()

### Community 27 - "Synthetic Series Fixture"
Cohesion: 0.50
Nodes (3): Synthetic time-series generator with trend, daily/weekly seasonality + noise., Return a pd.Series indexed by tz-aware UTC timestamps., synthetic_series()

## Knowledge Gaps
- **29 isolated node(s):** `list`, `description`, `editable`, `fiscalYearStartMonth`, `graphTooltip` (+24 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **20 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_pipeline()` connect `Artifact Storage` to `Config Loading`, `Registry Models`, `Model Ranking`, `Backtesting & Walk-Forward CV`?**
  _High betweenness centrality (0.142) - this node is a cross-community bridge._
- **Why does `RegistryRepo` connect `Registry Models` to `Diagnostics Endpoints`, `FastAPI App & Prometheus Export`, `Pipeline E2E Test`, `Artifact Storage`?**
  _High betweenness centrality (0.130) - this node is a cross-community bridge._
- **Why does `BaseForecaster` connect `Base Forecaster` to `Tabular Features & Forecast Result`, `LSTM Forecaster`, `LightGBM Forecaster`, `Prophet Forecaster`, `Forecaster Protocol`, `ARIMA Forecaster`, `ETS Forecaster`, `Holt-Winters Forecaster`, `Naive Forecaster`, `Seasonal Naive Forecaster`, `XGBoost Forecaster`?**
  _High betweenness centrality (0.095) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `RegistryRepo` (e.g. with `Base` and `Forecast`) actually correct?**
  _`RegistryRepo` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `BaseForecaster` (e.g. with `HoltWintersForecaster` and `ProphetForecaster`) actually correct?**
  _`BaseForecaster` has 10 INFERRED edges - model-reasoned connections that need verification._
- **What connects `list`, `description`, `editable` to the rest of the system?**
  _129 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Config Loading` be split into smaller, more focused modules?**
  _Cohesion score 0.061170212765957445 - nodes in this community are weakly interconnected._