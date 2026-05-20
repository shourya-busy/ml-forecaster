# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Dockerized multi-model server-resource forecaster for a Prometheus / LGTM stack. For each `(instance, metric, horizon)` it trains 10 algorithms in parallel, ranks them with MAE/RMSE/MAPE/sMAPE/R², persists winners + bands to Postgres, and re-exposes them as Prometheus gauges so Grafana can overlay forecasts on live panels.

## Common commands

The Makefile drives the docker-compose stack. Most work happens *inside* containers (Postgres + Redis are required at runtime), not on the host.

```bash
make build           # build images
make up              # start api + scheduler + 2x worker + postgres + redis
make migrate         # alembic upgrade head (uses the `migrate` one-shot service)
make seed            # seed synthetic series for fake-1, fake-2 (calls `forecaster-seed` in api)
make test            # runs `python -m pytest -q` inside the worker image
make lint            # `ruff check .` (host)
make reset           # down -v (DROPS pgdata + model-store volumes)
make gpu             # compose up with docker-compose.gpu.yml overlay
make demo            # adds prom + grafana via `--profile demo`
```

Run a single test inside the worker image:

```bash
docker compose run --rm worker python -m pytest -q tests/unit/test_ranking.py::test_compute_composite
```

Run tests on the host (requires `.venv` with `pip install -e .[dev]`):

```bash
.venv/bin/pytest -q tests/unit/test_metrics.py
```

`conftest.py` injects `src/` onto `sys.path` and points `FORECASTER_CONFIG_DIR` at the repo's `config/` dir, so tests work without installing the package.

## Architecture (the parts that need cross-file reading)

Six services, four code surfaces:

- **api** (`src/forecaster/api/`) — FastAPI. `/metrics` is the Prometheus exposition built per-scrape from the DB (`prometheus_export.py`); `/runs`, `/rankings`, `/forecasts`, `/models`, `/config`, `/diagnostics`, `/healthz` are REST routes under `routes/`.
- **scheduler** (`src/forecaster/scheduling/scheduler.py`) — APScheduler. One cron job per horizon; each fires `jobs.fan_out` which enqueues Celery `forecaster.train` tasks for every discovered `(instance, metric)` for that horizon.
- **worker** (`src/forecaster/training/tasks.py` → `pipeline.py` → `parallel.py`) — Celery. One task per `(instance, metric, horizon)`. Inside the task, `train_all_algos` fans out across a `ProcessPoolExecutor` so a misbehaving algorithm can't crash its siblings; children return pickled bytes + scores, parent persists.
- **registry** (`src/forecaster/registry/`) — SQLAlchemy 2.0 models + `RegistryRepo` + Alembic migrations under `migrations/versions/`. Artifacts go to `VolumeArtifactStore` (the `model-store` volume); only paths land in Postgres.

### Two extension points worth understanding before touching anything

**`Forecaster` protocol** (`src/forecaster/models/base.py`) — every algorithm implements `fit / predict / predict_interval / lookback_required / save / load`. Subclass `BaseForecaster` to inherit residual-based prediction intervals; override `predict_interval` if the model has a native one. Models must be picklable (children return `pickle.dumps(model)` to the parent).

Adding an algorithm is a 3-step change, no central dispatch to edit:
1. Create `src/forecaster/models/<name>.py`, decorate the class with `@register("<name>")`.
2. Add `from . import <name>` to `src/forecaster/models/__init__.py` so the decorator runs.
3. Add `"<name>"` to `algorithms.enabled` in `config/default.yaml` (and optionally `algorithms.defaults.<name>` hyperparams, and per-metric shortlists under `algorithms.per_metric`).

**`TSDataSource`** (`src/forecaster/data/base.py`) — Prometheus and Mimir both implement it; `make_data_source` picks one based on `data_sources.active`. Switching data backends is a config change.

### Config loading is the source of truth

`src/forecaster/config/loader.py` deep-merges four YAMLs from `FORECASTER_CONFIG_DIR` in order: `default.yaml`, `data_sources.yaml`, `targets.yaml`, `exposition.yaml`. Then it overlays env-var overrides using `FORECASTER__<SECTION>__<KEY>` (double-underscore delimiter, lowercased path), then validates as a Pydantic `Settings`. The result is cached at module level; `reload_settings()` (also bound to SIGHUP on api + scheduler) invalidates it.

When changing config behavior, edit the YAML *and* the Pydantic schema in `config/schema.py` together — unknown keys are rejected.

### Cardinality control lives in `exposition.yaml`

`/metrics` rebuilds a fresh `CollectorRegistry` per scrape from DB state. Each series family (`forecast_best_value`, `forecast_value` per-model, `forecast_model_score`, training timings) is gated by a boolean under `exposition.emit`. Defaults turn off `per_model_*` series — flipping them all on for 300 servers × 3 metrics × 10 models is ~120k series. Change the flag, not the route code.

### Per-metric algorithm shortlists

`algorithms.per_metric.{cpu,mem,disk}` in `default.yaml` narrows which algorithms train for each metric (must be a subset of `algorithms.enabled`). Missing metrics fall back to the full enabled list. Rationale is in the YAML comments; preserve it when editing.

## Conventions

- Python 3.11+, src-layout, `pythonpath = ["src"]` in pyproject (so imports are `forecaster.*`, not `src.forecaster.*`).
- Ruff is the only linter: `ruff check .`. Config in `pyproject.toml` (`select = [E,F,W,I,UP,B,SIM]`, `ignore = [E501]`, line-length 100).
- Database URL, broker URLs, log level, `FORECASTER_USE_CUDA`, and `ARTIFACT_DIR` come from env vars (see `.env.example` and the `x-app-env` anchor in `docker-compose.yml`). Inside containers, config lives at `/app/config` and artifacts at `/var/lib/forecaster/models`.
- Migrations: `alembic -c /app/config/alembic.ini upgrade head` (the `migrate` compose service runs this; `RegistryRepo.create_schema()` exists for tests only).
- Tests are split into `tests/unit/` (no Postgres needed) and `tests/integration/` (SQLite-backed pipeline / API smoke tests using `tests/fixtures/synthetic_series.py`).

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
