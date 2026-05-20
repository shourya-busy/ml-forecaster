# Grafana overlay recipe

The forecaster's `/metrics` endpoint exposes Prometheus gauges with a
`ts` label that carries the forecast timestamp as an ISO-8601 string.
A direct PromQL overlay is awkward because Prometheus stores each
sample at scrape time, not at the forecast's future time.

## Approach A — point-in-future overlay (simplest)

Add a second query to your existing CPU/MEM/DISK panel:

```promql
forecast_best_value{instance="$instance", metric="cpu", horizon="medium", bound="point"}
```

This gives you a horizontal line per future timestamp. Use this when you
want a quick "the model says X for the next 24h" annotation.

## Approach B — recording-rule rewrite (Grafana-friendly)

Add to your Prometheus recording rules:

```yaml
groups:
  - name: forecaster_rewrites
    interval: 30s
    rules:
      - record: forecast:cpu:point
        expr: forecast_best_value{metric="cpu", bound="point"}
      - record: forecast:cpu:lower
        expr: forecast_best_value{metric="cpu", bound="lower"}
      - record: forecast:cpu:upper
        expr: forecast_best_value{metric="cpu", bound="upper"}
```

Then overlay `forecast:cpu:point` on your live CPU panel and use the
"Fill Below To" plot option with `forecast:cpu:lower` for the band.

## Approach C — JSON via Infinity datasource

If you want the time axis to literally extend into the future, use the
Grafana Infinity datasource and point it at:

```
http://forecaster-api:8000/forecasts?instance=$instance&metric=cpu&horizon=medium
```

Set `ts` as the time column, `point` as the value, and overlay it on
your existing Prometheus query in a Time series panel. This is the
cleanest visual: the prediction line literally extends past `now()`.

## Common gotchas

- The `ts` label is high-cardinality. If you only have Approach A
  enabled, lower `exposition.series_per_forecast` in config to keep
  Prometheus happy.
- Confidence bands require `exposition.emit.best_model_bounds: true`.
- `forecast_best_model_info{model=...} = 1` lets you display "Winner:
  lstm" as a stat panel.
