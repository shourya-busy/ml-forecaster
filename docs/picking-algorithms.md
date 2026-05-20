# Picking the right algorithm per metric

Two questions need to be separated:

1. **Which algorithms are *allowed* to compete for a given metric?**
2. **Once they compete, is the winner trustworthy?**

The system handles (1) via per-metric shortlists in config, and gives you the
tools to answer (2) via the diagnostics surface.

## How auto-selection works today

Every training run, for one `(server, metric, horizon)`:

1. The pipeline fetches 30 days of history.
2. It trains each algorithm in the shortlist (see below) in parallel.
3. Each algorithm is **walk-forward cross-validated** with K folds and scored
   on MAE, RMSE, MAPE, sMAPE, R² (see
   `src/forecaster/evaluation/metrics.py`).
4. Scores are min-max normalised within the run, direction-flipped so larger
   is always better, and combined with configurable weights into a single
   composite (see `src/forecaster/evaluation/ranking.py`).
5. The model with the highest composite is marked `is_best=True`; its
   forecast lands in `forecast_best_value` on `/metrics`.

The shortlist of contenders is the lever you have to influence this.

## Per-metric shortlists

`config/default.yaml`:

```yaml
algorithms:
  enabled: [naive, seasonal_naive, arima, ets, holt_winters, prophet,
            xgboost, lightgbm, lstm, nbeats]
  per_metric:
    cpu:  [seasonal_naive, holt_winters, prophet, xgboost, lightgbm, lstm, nbeats]
    mem:  [naive, arima, ets, prophet, xgboost, lstm]
    disk: [naive, arima, prophet, holt_winters]
```

Rationale for the shipped defaults:

| Metric | Shape | Drop                              | Keep                                          |
|--------|-------|-----------------------------------|-----------------------------------------------|
| CPU    | Bursty, strong daily/weekly seasonality | Plain `naive`, `arima`, `ets` (poor on bursty signals) | Seasonal + ML + DL models |
| MEM    | Slowly trending, mild seasonality | DL overkill, `seasonal_naive` over-fits | Trend & smoothing models + a couple of capable ML models |
| DISK   | Near-monotonic growth | All seasonal models (`seasonal_naive`, `holt_winters` seasonal, DL) | Trend-capable: `naive` (carry-forward), `arima`, `prophet`, `holt_winters` with trend-only |

Rules enforced at startup (`config/schema.py::AlgorithmConfig`):

- Every entry in `per_metric[m]` must be in `algorithms.enabled`.
- Every entry must be a registered algorithm (no typos).
- No duplicates inside a shortlist.
- An *empty* list raises — to "use all enabled algos for this metric",
  delete the key instead of giving it `[]`.

An unconfigured metric falls back to the full `enabled` list.

## Verifying the choice

Three diagnostics endpoints + a Grafana dashboard.

### `GET /diagnostics/winners`

```json
[
  {
    "instance": "fake-1", "metric": "cpu", "horizon": "medium",
    "current_winner": "lstm",
    "previous_winner": "prophet",
    "winner_since": "2026-05-19T09:00:00+00:00",
    "unique_winners_recent": 2,
    "recent_window_runs": 10,
    "current_top3": [{"rank": 1, "algo": "lstm", "composite": 0.91, ...}, ...]
  }
]
```

`unique_winners_recent == 1` over a meaningful window is what stability looks
like. `== K` means the model is flapping run-to-run; usually a sign that two
algorithms are roughly tied on composite score.

### `GET /diagnostics/winner-history?instance=&metric=&horizon=&limit=`

Oldest-first list of who won each run. Useful for spotting "we used to be
stable on Prophet, then switched to LSTM three weeks ago — what changed?"

### `GET /diagnostics/score-history?instance=&metric=&horizon=&algo=&score=&limit=`

Time series of any algo's score on any of the 5 metrics. Plot one algo's
MAE across the last 50 runs and look for monotonic growth (drift) or a
sudden step (data anomaly upstream).

### Grafana dashboard

`deploy/grafana/dashboards/forecaster-diagnostics.json` — import once, then
pick `$instance / $metric / $horizon / $score` from the variables at the top.
Four panels:

1. **Current winners** (table) — driven by `forecaster_winner`.
2. **Winner stability** (stat) — `forecaster_winner_unique_recent`. Green ≤1,
   amber 2–3, red ≥4.
3. **Score by model — latest run** (bar gauge) — `forecast_model_score`.
4. **Score history per model** (time-series) — same metric over time.

The Prom series live behind `emit.diagnostics` in `config/exposition.yaml`;
turn it off if cardinality becomes a concern.

## Troubleshooting: "the wrong algorithm is winning"

A quick recipe:

1. **Open `/diagnostics/winners`.** Is `unique_winners_recent` > 1? If yes,
   the ranking is flapping — the composites are too close to call. Tighten
   the shortlist or adjust `ranking.weights` to emphasise the metric you
   care about most (RMSE for outlier-sensitivity, MAPE for percentage error).
2. **Open `/diagnostics/score-history`** for the algo you *think* should be
   winning. Is its error growing? That algo's drifting — usually means
   either its hyperparams need tuning (`algorithms.defaults[algo]`) or the
   signal has changed shape and the algo isn't suited anymore.
3. **Open `/diagnostics/winner-history`.** Did the winner switch around a
   specific date? Cross-reference with a deployment or capacity event on
   the box.
4. **Cut the shortlist.** If `seasonal_naive` keeps winning on a DISK series
   because of one weird week of seasonality, remove it from `per_metric.disk`
   — there's no point letting it compete on a metric where its assumptions
   don't apply.
5. **Send `SIGHUP` to the api** (or hit `POST /config/reload`) and trigger a
   new run with `POST /runs`. Re-check `/diagnostics/winners`.

## When you need stronger control

This iteration intentionally does **not** implement:

- **Algorithm pinning** — locking a specific algo for a `(server, metric,
  horizon)` regardless of ranking. The DB schema is general enough to add a
  `target_overrides` table later.
- **Stability rules (hysteresis)** — only switch winner if the challenger
  beats the incumbent by X% composite. A future iteration would put this
  behind a `ranking.hysteresis_pct` knob.

If either becomes necessary, the diagnostics views above are the right place
to confirm it from data before adding the code.
