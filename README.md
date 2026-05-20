# forecaster

Dockerized multi-model server-resource forecaster for a Prometheus / LGTM stack.

Trains 10 forecasting algorithms in parallel per `(server, metric, horizon)`, ranks
them with 5 standard metrics (MAE, RMSE, MAPE, sMAPE, R²), and re-exposes the
results in Prometheus exposition format so you can scrape and overlay forecasts
on existing Grafana panels.

## Quick start

```bash
make build
make up          # api, scheduler, 2x worker, postgres, redis
make migrate     # apply alembic migrations
make seed        # seed synthetic series for two fake instances
curl localhost:8000/healthz
curl localhost:8000/metrics | head
make preflight   # release-readiness check
```

**Deploying to a real LGTM host?** Follow [`docs/setup-and-review.md`](docs/setup-and-review.md) —
end-to-end release runbook for `/var/www/ml` on Linux, including a step-by-step
verification of every feature in this repo.

## Architecture

See [`docs/architecture.md`](docs/architecture.md). Six services in compose:

- `api` — FastAPI; serves `/metrics` (Prometheus exposition) and REST.
- `scheduler` — APScheduler; enqueues training jobs per horizon cron.
- `worker` — Celery; trains all 10 algorithms in parallel for one `(server, metric, horizon)`.
- `postgres` — registry: runs, rankings, latest forecasts.
- `redis` — Celery broker + scheduler locks.
- `model-store` — volume for serialized model artifacts.

GPU is opt-in:

```bash
make gpu   # uses docker-compose.gpu.yml
```

## Configuration

Everything lives under `config/`:

| File                  | Purpose                                                |
|-----------------------|--------------------------------------------------------|
| `default.yaml`        | algorithms, horizons, lookbacks, ranking weights       |
| `data_sources.yaml`   | Prometheus / Mimir endpoints, auth, timeouts           |
| `targets.yaml`        | which servers × metrics to forecast (static or PromQL) |
| `exposition.yaml`     | which series the `/metrics` endpoint emits             |

Env-var overrides supported with `FORECASTER__<SECTION>__<KEY>` style.

## Adding an algorithm

See [`docs/adding-an-algorithm.md`](docs/adding-an-algorithm.md). Implement the
`Forecaster` protocol in `src/forecaster/models/`, decorate with
`@register("name")`, list it under `algorithms.enabled` in `default.yaml`.

## Grafana overlay

See [`docs/grafana-overlay.md`](docs/grafana-overlay.md). The TL;DR is to query
`forecast_best_value{instance=~"$instance", metric="cpu", horizon="medium", bound="point"}`
on the same panel as your live `netdata_cpu_...` query.
