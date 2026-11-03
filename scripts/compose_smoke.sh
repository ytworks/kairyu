#!/usr/bin/env bash
# Compose integration test (goal G3 gates C1-C3, gate F1d): readiness,
# completion, SSE, session affinity via metrics, replica kill -> eject ->
# prober recovery, and the deployed gateway-to-replica OTel trace.
#
# The verdict depends only on the code under test: every wait observes a state
# or an event, and its bound (HANG_BOUND_S) only detects a hang.
#
# Usage: compose_smoke.sh [--setup-only | --no-setup]
#   (none)        build the images, then run the test
#   --setup-only  build the images (the networked setup; CI retries it) and exit
#   --no-setup    run the test on the images built by --setup-only; never
#                 build or pull
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$REPO_ROOT/deploy/compose/docker-compose.yaml"
COMPOSE_VALIDATOR="$REPO_ROOT/scripts/validate_compose_binds.py"
TRACE_VERIFIER="$REPO_ROOT/scripts/verify_otel_trace.py"
BASE_URL="${BASE_URL:-http://localhost:8000}"
# Far above every expected wait (seconds at most); reaching it means a hang.
HANG_BOUND_S=300
MODE="${1:-}"
case "$MODE" in
  "" | --setup-only | --no-setup) ;;
  *) echo "usage: $0 [--setup-only | --no-setup]" >&2; exit 2 ;;
esac
# Docker BuildKit inherits the invoking process's descriptor ceiling. The
# default 1024 on some developer hosts is insufficient for uv bytecode
# compilation; inability to raise it is harmless because the build still fails
# normally rather than being reported as a passing test.
ulimit -n 65536 2>/dev/null || true
compose() { docker compose -f "$COMPOSE_FILE" "$@"; }

if [[ "$MODE" != --setup-only ]]; then
  echo "== validate compose binds =="
  uv run --frozen --project "$REPO_ROOT" --no-dev \
    python "$COMPOSE_VALIDATOR" "$COMPOSE_FILE"
fi
if [[ "$MODE" != --no-setup ]]; then
  echo "== setup: build images =="
  compose build
fi
[[ "$MODE" != --setup-only ]] || exit 0

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/kairyu-compose-smoke.XXXXXX")"
BODY="$WORK_DIR/body"
TRACE_HEADERS="$WORK_DIR/otel-headers"
TRACE_BODY="$WORK_DIR/otel-body"
TRACE_LOGS="$WORK_DIR/otel-logs"

cleanup() {
  compose down --volumes --remove-orphans >/dev/null 2>&1 || true
  rm -rf "$WORK_DIR"
}

fail() { echo "SMOKE FAIL: $1" >&2; compose logs --tail 30 gateway >&2 || true; exit 1; }

hang() {
  echo "SMOKE HANG: $1 did not happen within ${HANG_BOUND_S}s (a hang bound, not a speed limit)" >&2
  compose logs --tail 30 gateway >&2 || true
  exit 1
}

wait_until() { # wait_until <description> <command...>: poll a state until it holds
  local description=$1 started=$SECONDS
  shift
  until "$@"; do
    ((SECONDS - started < HANG_BOUND_S)) || hang "$description"
    sleep 1
  done
  # Diagnostic only; elapsed time never decides the verdict.
  echo "   observed: $description (after $((SECONDS - started))s)"
}

ready() {
  local body
  body=$(curl -sf --max-time 30 "$BASE_URL/readyz" 2>/dev/null) && [[ "$body" == *'"ready"'* ]]
}

chat() { # chat <user> -> http status
  curl -s -o "$BODY" -w '%{http_code}' -X POST "$BASE_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"llama\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"user\":\"$1\"}"
}

metric() { curl -s --max-time 30 "$BASE_URL/metrics" | grep -F "$1" | awk '{print $NF}' | head -1; }

replica_1_health_is() { # replica-1 is pool replica "0"
  [[ "$(metric 'kairyu_replica_healthy{pool="llama",replica="0"}')" == "$1" ]]
}

trap cleanup EXIT

echo "== up =="
compose up -d --no-build --pull never

echo "== readiness =="
wait_until "gateway readiness" ready

echo "== non-stream completion =="
[[ "$(chat alice)" == 200 ]] || fail "completion returned $(cat "$BODY")"
grep -q '"chat.completion"' "$BODY" || fail "unexpected completion body"

echo "== SSE stream =="
curl -sN -X POST "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama","messages":[{"role":"user","content":"hello"}],"stream":true}' \
  | grep -q 'data: \[DONE\]' || fail "SSE stream missing [DONE]"

echo "== session affinity (gate C1) =="
for _ in 1 2 3 4; do [[ "$(chat alice)" == 200 ]] || fail "affinity request failed"; done
affinity=$(metric 'kairyu_pool_decisions_total{pool="llama",reason="session_affinity"}')
[[ "${affinity%.*}" -ge 4 ]] || fail "expected >=4 session_affinity decisions, got $affinity"

echo "== kill replica-1: eject then zero 5xx (gate C2) =="
# Ejection is a transition: replica-1 must be validated (healthy) before the
# kill. The prober validates it independently of the requests above.
wait_until "prober validating replica-1 before the kill" replica_1_health_is "1.0"
compose kill replica-1
docker wait "$(compose ps -aq replica-1)" >/dev/null
if ! convergence_status=$(curl -sS -o "$BODY" -w '%{http_code}' \
  -X POST "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama","messages":[{"role":"user","content":"x"}]}'); then
  fail "request during ejection convergence could not reach the gateway"
fi
case "$convergence_status" in
  200|502) ;;
  *) fail "request during ejection convergence returned $convergence_status" ;;
esac
wait_until "gateway ejecting replica-1 after it stopped" replica_1_health_is "0.0"
for i in $(seq 1 10); do
  [[ "$(chat "user-$i")" == 200 ]] || fail "request after ejection returned non-200"
done

echo "== restart replica-1: prober recovery (gate C2) =="
compose start replica-1
wait_until "prober restoring replica-1" replica_1_health_is "1.0"
wait_until "gateway readiness after recovery" ready

echo "== deployed OTel gateway-to-replica trace (gate F1d) =="
TRACE_PROMPT="otel-prompt-canary-f1d-2140f56d"
trace_status=$(curl -sS -D "$TRACE_HEADERS" -o "$TRACE_BODY" -w '%{http_code}' \
  -X POST "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"llama\",\"messages\":[{\"role\":\"user\",\"content\":\"$TRACE_PROMPT\"}]}")
[[ "$trace_status" == 200 ]] || fail "OTel canary returned HTTP $trace_status"
trace_request_id=$(
  awk 'tolower($1) == "x-request-id:" {gsub("\r", "", $2); print $2}' \
    "$TRACE_HEADERS" | tail -1
)
[[ -n "$trace_request_id" ]] || fail "OTel canary response omitted X-Request-ID"
# A graceful stop finishes every in-flight request and exits each process, and
# Docker drains a stopped container's output into its log. Every span is then
# in the logs, so they are read and verified exactly once.
compose stop --timeout "$HANG_BOUND_S"
for container in $(compose ps -aq); do
  exit_code=$(docker inspect --format '{{.State.ExitCode}}' "$container")
  echo "   stopped: $container exit code $exit_code"
  [[ "$exit_code" != 137 ]] || hang "graceful exit of container $container"
done
compose logs --no-color --no-log-prefix >"$TRACE_LOGS"
uv run --frozen --project "$REPO_ROOT" --no-dev \
  python "$TRACE_VERIFIER" \
    --logs "$TRACE_LOGS" \
    --request-id "$trace_request_id" \
    --response-body "$TRACE_BODY" \
    --forbidden "$TRACE_PROMPT" \
  || fail "deployed OTel trace did not satisfy F1d"

echo "SMOKE PASS"
