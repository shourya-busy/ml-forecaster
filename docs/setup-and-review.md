# Setup & review guide — releasing on the central LGTM server

End-to-end runbook for deploying the forecaster on a Linux server (the central
LGTM host) under `/var/www/ml`, then walking through every implemented feature
to confirm it works.

Read this in order on first deploy. After the initial setup, the **§5 Review
checklist** is what you come back to after every config change or upgrade.

> **Conventions used below**
> - All commands assume you're `cd /var/www/ml/ml-forecaster` unless stated.
> - Commands prefixed `$` run on the host shell; `psql>` run inside the
>   postgres container; `python>` inside an interactive Python shell.
> - "Expected" boxes describe what success looks like — diverge → see §7
>   Troubleshooting.

---

## 1. Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| OS | Ubuntu 22.04 / Debian 12 / RHEL 9 | Anything with a recent kernel + Docker |
| RAM | 8 GB | Worker images load Torch/Prophet; 16 GB recommended for >100 servers |
| Disk | 20 GB free | Postgres + model-store + image layers |
| Docker Engine | 24.0+ | `docker version` |
| Docker Compose plugin | 2.20+ | `docker compose version` (note: space, not hyphen) |
| Network | egress to your Prometheus / Mimir endpoint | typically internal |
| Ports | 8000 (api), 5432 (postgres), 6379 (redis) — bind to localhost in prod | bound to `0.0.0.0` by default in `docker-compose.yml`, see §6.3 |

Install Docker + Compose if missing:

```bash
$ curl -fsSL https://get.docker.com | sh
$ sudo systemctl enable --now docker
$ sudo usermod -aG docker "$USER"   # log out + back in
```

Verify:

```bash
$ docker version | head -2
$ docker compose version
```

---

## 2. Stage 1 — get the code onto the server

The project ships as a self-contained directory. Pick **one** transport.

### Option A: rsync from your workstation

```bash
# from your workstation
$ rsync -av --exclude '.venv' --exclude '__pycache__' \
        --exclude '.pytest_cache' --exclude '*.egg-info' \
        ./ml-forecaster/ root@lgtm-host:/var/www/ml/ml-forecaster/
```

### Option B: scp a tarball

```bash
# from your workstation
$ tar --exclude='.venv' --exclude='__pycache__' --exclude='*.egg-info' \
      -czf ml-forecaster.tgz ml-forecaster/
$ scp ml-forecaster.tgz root@lgtm-host:/tmp/
# on the LGTM host
$ sudo mkdir -p /var/www/ml && cd /var/www/ml
$ sudo tar xzf /tmp/ml-forecaster.tgz
$ sudo chown -R "$USER:$USER" /var/www/ml/ml-forecaster
```

### Option C: from a private git remote (recommended once you have one)

```bash
$ sudo mkdir -p /var/www/ml && sudo chown "$USER:$USER" /var/www/ml
$ cd /var/www/ml
$ git clone git@your-git-host:org/ml-forecaster.git
```

Verify layout:

```bash
$ cd /var/www/ml/ml-forecaster
$ ls
config  docker-compose.gpu.yml  docker-compose.yml  Dockerfile  docs
Makefile  pyproject.toml  README.md  scripts  src  tests  deploy  ...
```

---

## 3. Stage 2 — configure for your LGTM stack

Four YAMLs in `config/` are the entire dial board. **Always edit YAML, never
the code.** Env vars (see `.env.example`) override individual keys when you
need short-lived overrides.

### 3.1 Point the data source at your Prometheus

```bash
$ vim config/data_sources.yaml
```

```yaml
data_sources:
  active: prometheus_default
  endpoints:
    prometheus_default:
      kind: prometheus
      base_url: "http://prometheus.internal:9090"  # ← your URL
      timeout_seconds: 30
      verify_tls: true
      # optional:
      # bearer_token: "..."
      # basic_auth_user: "..."
      # basic_auth_password: "..."
    mimir_central:
      kind: mimir
      base_url: "http://mimir.internal:9009/prometheus"
      tenant_id: "anonymous"      # set once you enable multi-tenancy
      timeout_seconds: 60
```

When you later add Mimir, flip `data_sources.active: mimir_central` and
`make restart` (no code change).

### 3.2 Discover servers from Prometheus labels

```bash
$ vim config/targets.yaml
```

```yaml
targets:
  discovery: promql
  # adjust to match how Netdata appears in YOUR Prometheus job names
  discovery_query: 'group by (instance) (up{job="netdata"})'
  instance_label: instance
  static_instances: []   # used only when discovery=static
```

### 3.3 Verify your PromQL queries return per-instance series

```bash
$ vim config/default.yaml
```

Under `metrics_to_forecast.queries`, edit each PromQL so it produces a series
per server. **Test each query against your Prometheus first** (see §5.2.2).
The shipped defaults assume Netdata's `netdata_cpu_*` / `netdata_mem_*` /
`netdata_disk_*` metric names — adjust if you renamed jobs / dropped
dimensions.

### 3.4 Per-metric algorithm shortlists (optional, recommended)

The shipped defaults already filter for metric shape (CPU: seasonal + ML;
MEM: trend models; DISK: monotonic-friendly). See `docs/picking-algorithms.md`
for the rationale; tweak if you have a custom metric.

### 3.5 Cardinality

Open `config/exposition.yaml`. With 300 servers you may want to flip
`per_model_forecast: false` (saves ~27k Prometheus series). The dashboard
still works either way — it reads from Postgres.

### 3.6 Environment file

```bash
$ cp .env.example .env
$ vim .env
```

For 300 servers, set `CELERY_CONCURRENCY=4` per worker and bump `replicas`
in `docker-compose.yml` under the `worker:` service to 3-4.

---

## 4. Stage 3 — build, migrate, and start

```bash
$ cd /var/www/ml/ml-forecaster

# 1. Build images (worker image is ~2-3 GB due to Torch + Prophet)
$ make build

# 2. Bring the stack up (postgres, redis, api, scheduler, 2x worker)
$ make up

# 3. Apply DB migrations (the `migrate` one-shot service has already run via
#    depends_on: service_completed_successfully, but verify)
$ make migrate

# 4. Confirm everything is alive
$ make ps
```

**Expected** — `make ps` shows all services `running` and `(healthy)`:

```
NAME                  IMAGE                COMMAND             STATUS
ml-forecaster-api-1   ...                  "forecaster-api"    Up 30s (healthy)
ml-forecaster-postgres-1   postgres:16-alpine ...              Up 35s (healthy)
ml-forecaster-redis-1      redis:7-alpine    "docker-entrypoint…" Up 35s (healthy)
ml-forecaster-scheduler-1  ...               "forecaster-schedul…" Up 30s
ml-forecaster-worker-1     ...               "forecaster-worker"  Up 30s
ml-forecaster-worker-2     ...               "forecaster-worker"  Up 30s
```

If any service is `unhealthy` or restarting, see §7.

---

## 5. Stage 4 — review checklist (verify every implemented feature)

Run these in order on first deploy and after every config / image upgrade.
Each numbered check has a command, an expected outcome, and a pointer to
which code path / file is being exercised.

A scripted version of the most common checks lives at
`scripts/preflight.sh` — run `bash scripts/preflight.sh` for a one-shot pass
after working through this list manually the first time.

### 5.1 Process & networking

#### 5.1.1 API liveness
```bash
$ curl -sf http://localhost:8000/healthz | jq
```
**Expected:** `{"status": "ok"}`. Exercises FastAPI app boot + signal handler
install (`src/forecaster/api/main.py`).

#### 5.1.2 DB readiness
```bash
$ curl -sf http://localhost:8000/readyz | jq
```
**Expected:** `{"status": "ready"}`. Hits Postgres via `RegistryRepo`.

#### 5.1.3 Postgres reachability
```bash
$ docker compose exec postgres psql -U forecaster -d forecaster -c '\dt'
```
**Expected:** 5 tables — `training_runs`, `run_metrics`, `model_artifacts`,
`forecasts`, `rankings`. Plus `alembic_version`. Verifies that
`migrations/versions/0001_initial.py` ran.

#### 5.1.4 Redis reachability
```bash
$ docker compose exec redis redis-cli ping
```
**Expected:** `PONG`.

#### 5.1.5 Scheduler is ticking
```bash
$ docker compose logs --tail=30 scheduler | grep -E "registered horizon|Job"
```
**Expected:** one `registered horizon=...` line per configured horizon.
Source: `src/forecaster/scheduling/scheduler.py`.

#### 5.1.6 Celery workers are listening
```bash
$ docker compose logs --tail=30 worker | grep -E "ready|celery@"
```
**Expected:** `celery@worker@... ready` from each worker replica.

### 5.2 Config + data source

#### 5.2.1 Effective config loads
```bash
$ curl -s http://localhost:8000/config | jq '.horizons | keys, .algorithms.enabled'
```
**Expected:** all 3 horizon keys (`short`, `medium`, `long`) and all 10 algorithms.
Exercises `config/loader.py` deep-merge + Pydantic validation.

#### 5.2.2 Prometheus query roundtrips
Test each PromQL query you put in `metrics_to_forecast` directly against your
Prometheus first, then via the forecaster's discovery:

```bash
# Direct against Prometheus
$ curl -sG http://prometheus.internal:9090/api/v1/query \
    --data-urlencode 'query=group by (instance) (up{job="netdata"})' | jq '.data.result | length'
```

**Expected:** the number of edge nodes Prometheus sees. If 0 → fix the
`job=...` filter in `targets.yaml`.

Then via the forecaster's data layer (instance discovery uses the same code
path as training fetches):

```bash
$ docker compose exec api python -c "
from forecaster.config.loader import get_settings
from forecaster.scheduling.jobs import discover_targets
print(len(discover_targets()), 'instances discovered')
print(discover_targets()[:5])
"
```

**Expected:** matches the Prometheus count above. Exercises
`data/prometheus_client.py::PrometheusClient.discover_instances`.

#### 5.2.3 Per-metric shortlists are honoured
```bash
$ curl -s http://localhost:8000/config | jq '.algorithms.per_metric'
```
**Expected:** the per-metric block from `default.yaml` (e.g. `cpu`, `mem`,
`disk` keys with lists of algos that are all in `algorithms.enabled`).

### 5.3 Model registry

#### 5.3.1 All 10 algorithms registered
```bash
$ curl -s http://localhost:8000/models | jq '.registered, (.registered | length)'
```
**Expected:** the sorted list of 10 algorithms and `10`.
Exercises `models/registry.py` + auto-registration in `models/__init__.py`.

#### 5.3.2 Enabled set matches your config
```bash
$ curl -s http://localhost:8000/models | jq '.enabled | length, .disabled_but_registered'
```
**Expected:** matches `algorithms.enabled` count from §5.2.1.

### 5.4 Synchronous training run end-to-end

This is the **most important single check** — it exercises everything: data
fetch, walk-forward backtest, all 10 algos, ranking, artifact persistence,
DB writes.

#### 5.4.1 Pick a real server name to test with
```bash
$ docker compose exec api python -c "
from forecaster.scheduling.jobs import discover_targets
print(discover_targets()[0])
"
```

Note the value; call it `$INSTANCE` below.

#### 5.4.2 Trigger a synchronous run (blocks until done — 30-120s)
```bash
$ INSTANCE='<your value>'
$ curl -s -X POST http://localhost:8000/runs/sync \
        -H 'content-type: application/json' \
        -d "{\"instance\":\"$INSTANCE\",\"metric\":\"cpu\",\"horizon\":\"medium\"}" | jq
```
**Expected:** `{"run_id": <int>, "instance": "...", "metric": "cpu", "horizon": "medium"}`.

#### 5.4.3 Inspect the run
```bash
$ RUN_ID=<from previous>
$ curl -s http://localhost:8000/runs/$RUN_ID | jq '{status, duration_seconds, error}'
```
**Expected:** `status: "completed"`, non-null `duration_seconds`, `error: null`.

#### 5.4.4 Verify the per-metric shortlist was applied
```bash
$ docker compose exec postgres psql -U forecaster -d forecaster -c \
    "SELECT algo, count(*) FROM run_metrics WHERE run_id=$RUN_ID AND fold=-1 GROUP BY algo;"
```
**Expected:** rows only for the algorithms listed in
`algorithms.per_metric.cpu` (or `algorithms.enabled` if no shortlist for CPU).
If you see `nbeats` when CPU shortlist excludes it → the shortlist isn't being
read. Re-run §5.2.3.

#### 5.4.5 Verify ranking & winner
```bash
$ curl -s "http://localhost:8000/rankings?instance=$INSTANCE&metric=cpu&horizon=medium" | jq '.[0] | {winning_algo, ranked: (.ranked | length)}'
```
**Expected:** a winning algo (from the shortlist) and `ranked` count matching
the algos trained.

#### 5.4.6 Verify forecasts persisted
```bash
$ curl -s "http://localhost:8000/forecasts?instance=$INSTANCE&metric=cpu&horizon=medium&only_best=true" | jq 'length'
```
**Expected:** the number of forecast points. For `horizon=medium` (24h @
5min), that's up to 288.

### 5.5 Persistence

#### 5.5.1 Artifact files on disk
```bash
$ docker volume inspect ml-forecaster_model-store --format '{{ .Mountpoint }}'
$ sudo ls /var/lib/docker/volumes/ml-forecaster_model-store/_data/cpu/medium/ | head
```
**Expected:** one directory per instance, each containing
`{algo}/run-{id}.pkl` files. Exercises `registry/store.py::VolumeArtifactStore`.

#### 5.5.2 DB rows
```bash
$ docker compose exec postgres psql -U forecaster -d forecaster -c \
    "SELECT
       (SELECT count(*) FROM training_runs) AS runs,
       (SELECT count(*) FROM run_metrics)   AS metrics,
       (SELECT count(*) FROM forecasts)     AS forecasts,
       (SELECT count(*) FROM rankings)      AS rankings,
       (SELECT count(*) FROM model_artifacts) AS artifacts;"
```
**Expected:** all > 0 after a successful run.

### 5.6 Scheduler + Celery fan-out

#### 5.6.1 List the registered cron jobs
```bash
$ docker compose exec scheduler python -c "
from forecaster.scheduling.scheduler import _build_scheduler
sched = _build_scheduler()
for j in sched.get_jobs():
    print(j.id, '|', j.trigger)
"
```
**Expected:** one `fan_out_<horizon>` per configured horizon, each with the
configured cron expression.

#### 5.6.2 Manually fan out (without waiting for the next tick)
```bash
$ docker compose exec scheduler python -c "
from forecaster.scheduling.jobs import fan_out
n = fan_out('medium')
print('enqueued', n, 'tasks')
"
```
**Expected:** `enqueued N tasks`, where N = instances × metrics. Tasks land in
Redis; workers should pick them up:

```bash
$ docker compose logs --tail=50 worker | grep -E "train_task|Received task|forecaster.train"
```
**Expected:** `Received task: forecaster.train` lines.

### 5.7 REST API surface

All endpoints under `/docs` (Swagger UI). Spot-check from CLI:

```bash
$ for path in / /healthz /readyz /metrics /runs /rankings /forecasts /models /config /diagnostics/winners; do
    code=$(curl -so /dev/null -w '%{http_code}' "http://localhost:8000$path")
    printf "%4s  %s\n" "$code" "$path"
  done
```
**Expected:** `200` for everything except `/` (`307` redirect to `/ui/`).

### 5.8 Diagnostics endpoints

These are the "is the right algorithm winning?" surface from
`docs/picking-algorithms.md`.

```bash
$ curl -s "http://localhost:8000/diagnostics/winners" | jq '.[0]'
$ curl -s "http://localhost:8000/diagnostics/winner-history?instance=$INSTANCE&metric=cpu&horizon=medium" | jq '. | length'
$ curl -s "http://localhost:8000/diagnostics/score-history?instance=$INSTANCE&metric=cpu&horizon=medium&score=mae" | jq '.[0]'
```

**Expected (after at least one completed run):**
- `/winners` returns one row per `(instance, metric, horizon)` with
  `current_winner`, `unique_winners_recent`, `recent_window_runs`, `current_top3`.
- `/winner-history` returns ≥1 entry, oldest-first.
- `/score-history` returns flat list of `{run_id, completed_at, algo, score, value}`.

### 5.9 Prometheus exposition

#### 5.9.1 /metrics is emitting
```bash
$ curl -s http://localhost:8000/metrics | grep -E '^(forecast_|forecaster_)' | head -30
```
**Expected:** lines like:
```
forecast_best_value{instance="...", metric="cpu", horizon="medium", bound="point", ts="..."} 42.3
forecast_best_model_info{instance="...", metric="cpu", horizon="medium", model="lstm"} 1.0
forecast_model_score{instance="...", metric="cpu", horizon="medium", model="ets", score="mae"} 1.23
forecaster_winner{instance="...", metric="cpu", horizon="medium", model="lstm"} 1.0
forecaster_winner_unique_recent{instance="...", metric="cpu", horizon="medium"} 1.0
```

If any family is missing → the corresponding `emit.*` flag in
`config/exposition.yaml` is off. Flip it on, `POST /config/reload`, re-scrape.

#### 5.9.2 Cardinality sanity check
```bash
$ curl -s http://localhost:8000/metrics | grep -c '^forecast_'
```
**Expected:** roughly `instances × metrics × horizons × bounds × emitted_models`.
For 300 servers × 3 metrics × 3 horizons × 3 bounds × 10 models × all flags on
that's ≈ 80k. If you see > 100k, you forgot to disable `per_model_*` in prod.

#### 5.9.3 Wire your central Prometheus to scrape the forecaster
Add to your Prometheus config (NOT the demo profile's one — your actual prod
Prometheus):

```yaml
scrape_configs:
  - job_name: forecaster
    metrics_path: /metrics
    scrape_interval: 30s
    static_configs:
      - targets: ["<lgtm-host>:8000"]
```

Reload Prometheus (`SIGHUP` or via its `/-/reload` endpoint). Confirm:

```bash
$ curl -sG http://prometheus.internal:9090/api/v1/query \
       --data-urlencode 'query=count(forecast_best_value)' | jq '.data.result'
```
**Expected:** a non-zero value.

### 5.10 UI dashboard

Open `http://<lgtm-host>:8000/` in a browser. The bare hostname auto-redirects
to `/ui/`. (For prod, route this behind your reverse proxy — see §6.4.)

Walk through each page and confirm:

| Page | What to verify |
|---|---|
| **Overview** | 8 stat cards populate with sane numbers; "Recent training runs" lists your test run; "Targets needing attention" is empty (or shows expected flapping targets) |
| **Targets** (`/ui/targets`) | One row per `(instance, metric, horizon)`; current winner pill matches §5.4.5; stability pill is `stable` on a fresh single-run target |
| **Target Detail** (click any row) | Forecast chart with band renders; latest ranking bar chart shows all trained algos; score-history line chart populates after ≥2 runs; "Trigger a new training run" button works (you'll see the new run on the Runs page) |
| **Runs** (`/ui/runs`) | Your run is there with `completed` pill; filters by instance/metric/horizon/status work |
| **Run Detail** (click a run id) | Per-algo composite + duration bar charts render; per-algo scores table populated; config snapshot JSON pretty-printed |
| **Models** (`/ui/models`) | Win-rate bar chart + stacked wins-by-metric chart; all 10 algos visible (those with 0 runs show `—`) |
| **Config** (`/ui/config`) | Active data source matches §3.1; per-metric shortlists table reflects YAML; "Reload config" button redirects back successfully |

### 5.11 Grafana overlay

(Demo Grafana via `make demo` — for prod, use your existing Grafana.)

1. Add Prometheus as a datasource pointing at your central Prometheus.
2. Import `deploy/grafana/dashboards/forecaster-diagnostics.json` — pick
   `$instance`, `$metric`, `$horizon` from the variables.
3. Confirm the four panels render: Current winners table, Winner stability
   stat, Score-by-model bar gauge, Score history timeseries.
4. Overlay forecasts on your live Netdata panels — recipe in
   `docs/grafana-overlay.md`.

### 5.12 Test suite

Optional but worth running once per upgrade:

```bash
$ make test
```
**Expected:** `47 passed, 5 skipped` (or whatever the current count is in the
worker image — skips are optional deps that aren't slow-loaded).

---

## 6. Stage 5 — production hygiene

### 6.1 Backups

The two stateful surfaces are Postgres and the `model-store` volume.

**Postgres** — daily logical dump, cron at 03:00:
```bash
$ sudo crontab -e
# add:
0 3 * * * cd /var/www/ml/ml-forecaster && docker compose exec -T postgres \
  pg_dump -U forecaster forecaster | gzip > /var/backups/forecaster-$(date +\%F).sql.gz
```

**model-store** — `tar` weekly. Less critical since models are recomputed on
the next retrain; mainly useful to avoid the warm-up gap after a disaster.

### 6.2 Log rotation

Compose logs to the docker JSON driver by default; cap them in
`/etc/docker/daemon.json`:

```json
{
  "log-driver": "json-file",
  "log-opts": { "max-size": "100m", "max-file": "5" }
}
```

Restart docker after editing: `sudo systemctl restart docker`.

### 6.3 Bind ports to localhost only

In production the stack should not expose Postgres/Redis on `0.0.0.0`. Edit
`docker-compose.yml` and prefix `127.0.0.1:`:

```yaml
postgres:
  ports:
    - "127.0.0.1:5432:5432"
redis:
  ports:
    - "127.0.0.1:6379:6379"
api:
  ports:
    - "127.0.0.1:8000:8000"   # then proxy via nginx — see 6.4
```

### 6.4 Reverse proxy + TLS (nginx example)

```nginx
server {
  listen 443 ssl http2;
  server_name forecaster.your-domain.internal;
  ssl_certificate     /etc/letsencrypt/live/.../fullchain.pem;
  ssl_certificate_key /etc/letsencrypt/live/.../privkey.pem;

  location / {
    proxy_pass         http://127.0.0.1:8000;
    proxy_set_header   Host              $host;
    proxy_set_header   X-Real-IP         $remote_addr;
    proxy_set_header   X-Forwarded-Proto $scheme;
    proxy_read_timeout 60s;
  }
}
```

### 6.5 systemd boot integration (optional)

```ini
# /etc/systemd/system/ml-forecaster.service
[Unit]
Description=ml-forecaster docker stack
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/var/www/ml/ml-forecaster
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down

[Install]
WantedBy=multi-user.target
```

```bash
$ sudo systemctl daemon-reload
$ sudo systemctl enable --now ml-forecaster
```

### 6.6 Resource limits

Add per-service `deploy.resources` blocks if you co-host with other services.
A safe starting point for the 300-server case:

```yaml
worker:
  deploy:
    resources:
      limits:   { cpus: '4',   memory: '6G' }
      reservations: { cpus: '1', memory: '2G' }
```

### 6.7 Move Docker's data-root onto your big data disk

Docker stores images, container layers, and volumes under `/var/lib/docker`
by default — which on many cloud images sits on a small root partition.
The worker image alone is ~2 GB (CPU-only) or ~5-6 GB (GPU build), so it's
worth moving Docker's storage onto whichever disk you provisioned for data.

Check what's where first:

```bash
$ df -h /var/lib/docker /var/www
$ docker system df          # how much docker is using
```

If the root partition is tight and `/var/www` (or wherever your data disk
is mounted) has space, relocate:

```bash
# 1. Stop the stack and the docker daemon
$ cd /var/www/ml/ml-forecaster && make down
$ sudo systemctl stop docker docker.socket

# 2. Create the new data-root and copy what's there
$ sudo mkdir -p /var/www/docker
$ sudo rsync -aP /var/lib/docker/ /var/www/docker/

# 3. Tell the daemon to use it (merge with any existing settings, e.g. log opts)
$ sudo tee /etc/docker/daemon.json >/dev/null <<'JSON'
{
  "data-root": "/var/www/docker",
  "log-driver": "json-file",
  "log-opts": { "max-size": "100m", "max-file": "5" }
}
JSON

# 4. Start docker, verify, then remove the old tree once you've confirmed
$ sudo systemctl start docker
$ docker info | grep -i "docker root dir"   # should print /var/www/docker
$ docker compose -f /var/www/ml/ml-forecaster/docker-compose.yml ps
# only after the stack is happy:
$ sudo rm -rf /var/lib/docker
```

---

## 7. Troubleshooting

| Symptom | First thing to check | Where |
|---|---|---|
| `make up` says "image build failed" | `make build` in the foreground; usually a network proxy / pip mirror issue. Set `HTTP_PROXY` / `HTTPS_PROXY` env vars before `make build`. | host shell |
| `make build` aborts with `no space left on device` while extracting `nvidia/cu*/lib/...` | PyTorch's default PyPI wheel bundles ~3-4 GB of CUDA libs. The Dockerfile now defaults to the CPU-only torch index (`TORCH_INDEX=cpu`). To recover: `docker system prune -af --volumes && docker builder prune -af`, pull latest code, then `make build` again. If your root partition is small but you have a bigger data disk, also move docker's data-root onto it (see §6.7). | `Dockerfile`, `/etc/docker/daemon.json` |
| API healthy but `/readyz` 500s | `docker compose logs postgres` — usually disk full or `pgdata` perms after a manual `chown` | postgres |
| `/runs/sync` returns 500, `error: "no data returned"` | Your PromQL doesn't match anything. Run the query directly against Prometheus (§5.2.2) and tune the filter | `config/default.yaml::metrics_to_forecast` |
| `/runs/sync` errors with `too few points after fetch+resample` | Prometheus retention is shorter than your `training.lookback_days`. Either lower `lookback_days` or extend retention | `config/default.yaml::training` |
| Worker logs show `arima` or `prophet` failing on a series | Expected for short / degenerate series — the pipeline marks that algo failed and skips it. Look at `/runs/{id}` JSON for the full traceback per algo | `pipeline.py` |
| Prom `/metrics` has 0 forecast series after a successful run | `exposition.emit.*` all off, OR the API and worker disagree on `DATABASE_URL`. Compare envs: `docker compose exec api env \| grep DATABASE` vs the same on `worker` | `.env` / `docker-compose.yml` |
| UI is blank / 500s | `docker compose logs api`. Most common: a YAML edit failed pydantic validation. The schema reject is logged at startup. | api logs |
| Scheduler fires but no runs appear | `docker compose logs scheduler` → confirm `enqueued N tasks`. If yes, check `docker compose logs worker` for "Received task". If workers don't see the task, Redis broker URLs disagree | both logs |
| Winner is flapping rapidly (Diagnostics page shows red) | This is information, not a bug — see `docs/picking-algorithms.md` "Troubleshooting" recipe | diagnostics |
| `pmdarima` import error in ARIMA | Python 3.13 dropped the wheel; we already gate it. Confirm `python --version` inside the worker (`docker compose exec worker python --version`). | worker image |
| `forecaster_winner` series missing while others present | `emit.diagnostics: false` in `exposition.yaml`. Toggle it on and `POST /ui/config/reload` or send SIGHUP to api | `config/exposition.yaml` |
| GPU override `make gpu` fails | Missing nvidia-container-toolkit; install per nvidia's instructions and run `sudo systemctl restart docker` | host |

---

## 8. Upgrade procedure

```bash
$ cd /var/www/ml/ml-forecaster

# 1. Get the new code
$ git pull   # or rsync from your workstation

# 2. Review what changed touching schema or config defaults
$ git log --oneline -- src/forecaster/registry/migrations config/

# 3. Rebuild + recreate
$ make build
$ make up         # picks up new image
$ make migrate    # runs any new alembic revisions

# 4. Run the §5 review checklist again — at minimum:
$ bash scripts/preflight.sh

# 5. Reload config (no restart needed for YAML-only changes)
$ curl -X POST http://localhost:8000/config/reload
$ docker compose kill -s SIGHUP scheduler   # scheduler re-reads its crons
```

If a migration introduces a non-trivial column, the `migrate` service runs
inside compose with `--exit-code-from`; failure means **the api won't come
up** until you investigate (`docker compose logs migrate`).

---

## 9. Quick reference card

```
HOST PATHS
  Project root         /var/www/ml/ml-forecaster
  Config               /var/www/ml/ml-forecaster/config/
  Compose log          docker compose logs <service>
  Postgres volume      /var/lib/docker/volumes/ml-forecaster_pgdata/_data
  Model artifacts      /var/lib/docker/volumes/ml-forecaster_model-store/_data

URLS (replace localhost with lgtm-host)
  Dashboard            http://localhost:8000/
  Swagger / API docs   http://localhost:8000/docs
  Prom exposition      http://localhost:8000/metrics
  Health               http://localhost:8000/healthz
  Diagnostics          http://localhost:8000/diagnostics/winners

USEFUL COMMANDS
  Boot                 make up
  Stop                 make down
  Rebuild              make build
  Reset state (!)      make reset       # drops pgdata + model-store
  Run tests in worker  make test
  GPU mode             make gpu
  Preflight check      bash scripts/preflight.sh

SIGNALS
  Reload api config    docker compose kill -s SIGHUP api
  Reload scheduler     docker compose kill -s SIGHUP scheduler

ENV OVERRIDES
  FORECASTER__TRAINING__LOOKBACK_DAYS=14
  FORECASTER__DATA_SOURCES__ACTIVE=mimir_central
  FORECASTER__EXPOSITION__EMIT__PER_MODEL_FORECAST=false
```

---

## 10. After the first successful release

1. Run `bash scripts/preflight.sh` daily for the first week — it surfaces
   silent failures (stale runs, flapping rankings, schema drift) quickly.
2. Open the **Targets needing attention** card on the Overview tab — empty is
   the desired steady state.
3. Within 24h you should see a winner per `(server, metric, horizon)` in
   `/ui/targets`. Within 7 days the score-history charts on the per-target
   detail page become meaningful for drift detection.
4. Once you're confident, switch `data_sources.active` to Mimir
   (`mimir_central`) and `POST /config/reload`. Re-run §5.2.2 to confirm
   discovery still works against the new endpoint.
