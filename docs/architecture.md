# Architecture

```
        ┌──────────────────┐    enqueue    ┌────────────┐
        │  scheduler       │──────────────▶│  redis     │
        │  (APScheduler)   │               │  (broker)  │
        └────────┬─────────┘               └──────┬─────┘
                 │                                │
                 │ fan_out(horizon)               │ pop
                 ▼                                ▼
        ┌──────────────────┐    fetch     ┌────────────┐
        │  worker (Celery) │◀────────────▶│  data      │
        │  trains 10 algos │              │  source    │
        │  in parallel     │              │  (Prom /   │
        │                  │              │   Mimir)   │
        └────────┬─────────┘              └────────────┘
                 │ persist
                 ▼
        ┌──────────────────┐         ┌──────────────────┐
        │  postgres        │         │  model-store     │
        │  (registry)      │         │  (volume)        │
        └────────┬─────────┘         └────────────────┬─┘
                 │                                    │
                 │ read                               │ load (optional)
                 ▼                                    ▼
        ┌──────────────────────────────────────────────┐
        │  api (FastAPI)                               │
        │  ├─ /metrics    (Prometheus exposition)      │
        │  ├─ /runs       (trigger + list)             │
        │  ├─ /rankings   (latest ranks)               │
        │  ├─ /forecasts  (JSON, with bands)           │
        │  ├─ /models     (registry introspection)     │
        │  └─ /config     (effective config + reload)  │
        └────────┬─────────────────────────────────────┘
                 │ scrape /metrics
                 ▼
        ┌──────────────────┐         ┌──────────────────┐
        │  prometheus      │────────▶│  grafana panels  │
        └──────────────────┘         └──────────────────┘
```

## Two abstractions worth knowing

**`TSDataSource`** (`src/forecaster/data/base.py`) — Prometheus and Mimir
both implement this. Switching is a one-line config change.

**`Forecaster`** (`src/forecaster/models/base.py`) — every algorithm
implements `fit`, `predict`, `predict_interval`. Add a new algo by:

1. Implement the protocol in a new file under `src/forecaster/models/`.
2. Decorate the class with `@register("my_algo")`.
3. Add `"my_algo"` to `algorithms.enabled` in `default.yaml`.

That's it — no other code touches algorithms by name.

## Cardinality control

`exposition.yaml` toggles each series family. With every flag on you
emit ~120k series for 300 servers × 3 metrics × 10 models. Sensible
prod defaults turn off `per_model_*` series and rely on
`forecast_best_*` + `forecast_model_score`.
