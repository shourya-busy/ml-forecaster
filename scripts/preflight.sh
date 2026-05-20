#!/usr/bin/env bash
# preflight.sh — one-shot release-readiness checks for ml-forecaster.
#
# Runs through the §5 review checklist from docs/setup-and-review.md and
# prints a green / red line per check. Exits non-zero if any required check
# fails so it's safe to wire into CI / cron.
#
# Run from the project root:
#   bash scripts/preflight.sh
#
# Optional env vars:
#   API_URL          base URL of the api (default: http://localhost:8000)
#   PROBE_INSTANCE   instance name to use for the end-to-end probe (default:
#                    first instance discovered, or "fake-1" if discovery is empty)
#   SKIP_RUN         set to 1 to skip the synchronous training-run probe
#                    (it can take 30-120s)

set -u

API_URL="${API_URL:-http://localhost:8000}"
PROBE_INSTANCE="${PROBE_INSTANCE:-}"
SKIP_RUN="${SKIP_RUN:-0}"

pass=0
fail=0
warn=0

GREEN=$'\033[0;32m'
RED=$'\033[0;31m'
YELLOW=$'\033[0;33m'
DIM=$'\033[0;90m'
RESET=$'\033[0m'

ok()   { printf "  ${GREEN}✓${RESET} %s\n" "$1"; pass=$((pass+1)); }
bad()  { printf "  ${RED}✗${RESET} %s\n     ${DIM}%s${RESET}\n" "$1" "$2"; fail=$((fail+1)); }
warns(){ printf "  ${YELLOW}!${RESET} %s\n     ${DIM}%s${RESET}\n" "$1" "$2"; warn=$((warn+1)); }
hdr()  { printf "\n${YELLOW}▸ %s${RESET}\n" "$1"; }

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    bad "missing dependency: $1" "install $1 or adjust PATH"
    exit 2
  fi
}
need curl
need docker
need jq

# ----------------------------------------------------------------------------
hdr "Compose services"

if ! docker compose ps --status running --format '{{.Service}}' > /tmp/preflight.services 2>/dev/null; then
  bad "docker compose not reachable" "are you in the project root? is docker running?"
  exit 2
fi

for svc in postgres redis api scheduler worker; do
  if grep -qx "$svc" /tmp/preflight.services; then
    ok "service running: $svc"
  else
    bad "service not running: $svc" "docker compose logs $svc"
  fi
done

# ----------------------------------------------------------------------------
hdr "Liveness / readiness"

health=$(curl -fsS -m 5 "$API_URL/healthz" 2>/dev/null || true)
if [[ "$health" == *'"status":"ok"'* ]]; then
  ok "API /healthz responding"
else
  bad "API /healthz did not return ok" "is the api container up? curl $API_URL/healthz"
fi

ready=$(curl -fsS -m 5 "$API_URL/readyz" 2>/dev/null || true)
if [[ "$ready" == *'"status":"ready"'* ]]; then
  ok "API /readyz (postgres reachable)"
else
  bad "API /readyz failed" "postgres unreachable — docker compose logs postgres"
fi

if docker compose exec -T redis redis-cli ping 2>/dev/null | grep -q PONG; then
  ok "redis PONG"
else
  bad "redis not responding" "docker compose logs redis"
fi

# ----------------------------------------------------------------------------
hdr "Database schema"

tables=$(docker compose exec -T postgres psql -U forecaster -d forecaster -tAc \
  "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename;" 2>/dev/null)
expected_tables=(training_runs run_metrics model_artifacts forecasts rankings alembic_version)
all_present=1
for t in "${expected_tables[@]}"; do
  if echo "$tables" | grep -qx "$t"; then
    :
  else
    bad "missing table: $t" "did migrations run? make migrate"
    all_present=0
  fi
done
[[ $all_present -eq 1 ]] && ok "all expected tables present"

# ----------------------------------------------------------------------------
hdr "Config & data source"

cfg=$(curl -fsS "$API_URL/config" 2>/dev/null || true)
if [[ -n "$cfg" ]]; then
  algos=$(echo "$cfg" | jq -r '.algorithms.enabled | length' 2>/dev/null || echo 0)
  horizons=$(echo "$cfg" | jq -r '.horizons | length' 2>/dev/null || echo 0)
  if [[ "$algos" -gt 0 && "$horizons" -gt 0 ]]; then
    ok "/config: $algos algorithms enabled, $horizons horizons configured"
  else
    bad "/config: algorithms=$algos horizons=$horizons" "check config/default.yaml"
  fi

  per_metric=$(echo "$cfg" | jq -r '.algorithms.per_metric | keys | length')
  if [[ "$per_metric" -gt 0 ]]; then
    ok "/config: per-metric shortlists configured ($per_metric metrics)"
  else
    warns "/config: no per-metric shortlists" "every metric will train all enabled algos"
  fi
else
  bad "/config did not return JSON" "is the api booted?"
fi

# ----------------------------------------------------------------------------
hdr "Model registry"

models=$(curl -fsS "$API_URL/models" 2>/dev/null || true)
registered_count=$(echo "$models" | jq -r '.registered | length' 2>/dev/null || echo 0)
if [[ "$registered_count" -ge 10 ]]; then
  ok "/models: $registered_count algorithms registered"
else
  bad "/models: only $registered_count registered, expected ≥10" "are the model modules importing? check api logs"
fi

# ----------------------------------------------------------------------------
hdr "Instance discovery"

discovered=$(docker compose exec -T api python -c '
from forecaster.scheduling.jobs import discover_targets
try:
    insts = discover_targets()
    print(len(insts))
    for i in insts[:3]: print(i)
except Exception as e:
    print(f"ERROR: {e}")
' 2>/dev/null || true)
first_line=$(echo "$discovered" | head -1)

if [[ "$first_line" =~ ^[0-9]+$ ]] && [[ "$first_line" -gt 0 ]]; then
  ok "Prometheus discovery returned $first_line instance(s)"
  if [[ -z "$PROBE_INSTANCE" ]]; then
    PROBE_INSTANCE=$(echo "$discovered" | sed -n '2p')
  fi
elif [[ "$first_line" == "0" ]]; then
  warns "Prometheus discovery returned 0 instances" "check config/targets.yaml discovery_query"
  [[ -z "$PROBE_INSTANCE" ]] && PROBE_INSTANCE="fake-1"
else
  warns "discovery failed" "$first_line — check connectivity to Prometheus"
  [[ -z "$PROBE_INSTANCE" ]] && PROBE_INSTANCE="fake-1"
fi

# ----------------------------------------------------------------------------
hdr "Scheduler"

sched_logs=$(docker compose logs --tail=200 scheduler 2>/dev/null || true)
horizon_count=$(echo "$sched_logs" | grep -c "registered horizon=" || true)
if [[ "$horizon_count" -ge 1 ]]; then
  ok "scheduler registered $horizon_count horizon cron job(s)"
else
  bad "scheduler logs show no registered horizons" "docker compose logs scheduler"
fi

# ----------------------------------------------------------------------------
hdr "Celery worker"

w_logs=$(docker compose logs --tail=200 worker 2>/dev/null || true)
if echo "$w_logs" | grep -qE "celery@.*ready"; then
  ok "celery worker(s) reported ready"
else
  warns "no 'celery ... ready' marker in recent worker logs" "may have rolled off — bump --tail or check 'docker compose ps worker'"
fi

# ----------------------------------------------------------------------------
hdr "Prometheus exposition"

metrics_out=$(curl -fsS "$API_URL/metrics" 2>/dev/null || true)
if [[ -n "$metrics_out" ]]; then
  total_series=$(echo "$metrics_out" | grep -cE '^(forecast_|forecaster_)' || true)
  if [[ "$total_series" -gt 0 ]]; then
    ok "/metrics: $total_series forecast / forecaster series"
  else
    warns "/metrics: no forecast_* or forecaster_* series" "no runs persisted yet, or all emit.* flags off"
  fi
else
  bad "/metrics endpoint not returning" "docker compose logs api"
fi

# ----------------------------------------------------------------------------
hdr "UI dashboard"

ui_code=$(curl -so /dev/null -w '%{http_code}' "$API_URL/ui/")
case "$ui_code" in
  200) ok "/ui/ overview page returned 200" ;;
  *)   bad "/ui/ returned $ui_code" "docker compose logs api" ;;
esac

# Static asset reachable?
asset_code=$(curl -so /dev/null -w '%{http_code}' "$API_URL/ui/static/css/app.css")
case "$asset_code" in
  200) ok "/ui/static/css/app.css reachable" ;;
  *)   bad "/ui/ static assets returning $asset_code" "templates/static may not have shipped in the image" ;;
esac

# ----------------------------------------------------------------------------
if [[ "$SKIP_RUN" != "1" ]]; then
  hdr "Synchronous training-run probe (30-120s)"
  printf "  ${DIM}probing with instance='%s' metric=cpu horizon=medium${RESET}\n" "$PROBE_INSTANCE"

  payload=$(printf '{"instance":"%s","metric":"cpu","horizon":"medium"}' "$PROBE_INSTANCE")
  sync_out=$(curl -fsS -X POST "$API_URL/runs/sync" \
                  -H 'content-type: application/json' \
                  -d "$payload" 2>/dev/null || true)
  run_id=$(echo "$sync_out" | jq -r '.run_id // empty' 2>/dev/null)
  if [[ -n "$run_id" ]]; then
    ok "POST /runs/sync produced run_id=$run_id"
    detail=$(curl -fsS "$API_URL/runs/$run_id")
    status=$(echo "$detail" | jq -r '.status')
    err=$(echo "$detail" | jq -r '.error // empty')
    if [[ "$status" == "completed" ]]; then
      ok "run #$run_id status: completed"
    else
      bad "run #$run_id status: $status" "${err:-no error message}"
    fi
    fcount=$(curl -fsS "$API_URL/forecasts?instance=$PROBE_INSTANCE&metric=cpu&horizon=medium" | jq 'length')
    if [[ "$fcount" -gt 0 ]]; then
      ok "$fcount forecast points persisted"
    else
      bad "no forecasts persisted" "check pipeline logs"
    fi
    rcount=$(curl -fsS "$API_URL/rankings?instance=$PROBE_INSTANCE&metric=cpu&horizon=medium" | jq 'length')
    if [[ "$rcount" -gt 0 ]]; then
      winner=$(curl -fsS "$API_URL/rankings?instance=$PROBE_INSTANCE&metric=cpu&horizon=medium" | jq -r '.[0].winning_algo')
      ok "ranking persisted, winner: $winner"
    else
      bad "no rankings persisted" "check ranking.weights / training output"
    fi
  else
    bad "POST /runs/sync did not produce a run_id" "$sync_out"
  fi
else
  hdr "Synchronous training-run probe (SKIPPED via SKIP_RUN=1)"
fi

# ----------------------------------------------------------------------------
printf "\n"
printf "${GREEN}%d passed${RESET}, ${YELLOW}%d warnings${RESET}, ${RED}%d failed${RESET}\n" \
  "$pass" "$warn" "$fail"

if [[ "$fail" -gt 0 ]]; then
  printf "${RED}preflight: FAIL${RESET} — see §7 Troubleshooting in docs/setup-and-review.md\n"
  exit 1
fi
printf "${GREEN}preflight: OK${RESET}\n"
exit 0
