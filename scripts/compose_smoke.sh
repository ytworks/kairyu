#!/usr/bin/env bash
# Compose integration test (goal G3 gates C1-C3): readiness, completion, SSE,
# session affinity via metrics, replica kill -> eject -> prober recovery.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$REPO_ROOT/deploy/compose/docker-compose.yaml"
COMPOSE_VALIDATOR="$REPO_ROOT/scripts/validate_compose_binds.py"
BASE_URL="${BASE_URL:-http://localhost:8000}"
# Docker BuildKit inherits the invoking process's descriptor ceiling. The
# default 1024 on some developer hosts is insufficient for uv bytecode
# compilation; inability to raise it is harmless because the build still fails
# normally rather than being reported as a passing test.
ulimit -n 65536 2>/dev/null || true
compose() { docker compose -f "$COMPOSE_FILE" "$@"; }

cleanup() { compose down --volumes --remove-orphans >/dev/null 2>&1 || true; }

fail() { echo "SMOKE FAIL: $1" >&2; compose logs --tail 30 gateway >&2 || true; exit 1; }

wait_for() { # wait_for <url> <substring> <attempts>
  local url=$1 want=$2 attempts=$3 body
  for _ in $(seq 1 "$attempts"); do
    body=$(curl -sf "$url" 2>/dev/null) && [[ "$body" == *"$want"* ]] && return 0
    sleep 2
  done
  return 1
}

chat() { # chat <user> -> http status
  curl -s -o /tmp/smoke_body -w '%{http_code}' -X POST "$BASE_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"llama\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"user\":\"$1\"}"
}

metric() { curl -s "$BASE_URL/metrics" | grep -F "$1" | awk '{print $NF}' | head -1; }

wait_for_metric() { # wait_for_metric <metric selector> <value> <attempts>
  local selector=$1 want=$2 attempts=$3 value
  for _ in $(seq 1 "$attempts"); do
    value=$(metric "$selector") || value=
    [[ "$value" == "$want" ]] && return 0
    sleep 2
  done
  return 1
}

echo "== validate compose binds =="
uv run --frozen --project "$REPO_ROOT" --no-dev \
  python "$COMPOSE_VALIDATOR" "$COMPOSE_FILE"
trap cleanup EXIT

echo "== up =="
compose up -d --build --quiet-pull

echo "== readiness =="
wait_for "$BASE_URL/readyz" '"ready"' 60 || fail "gateway never became ready"

echo "== non-stream completion =="
[[ "$(chat alice)" == 200 ]] || fail "completion returned $(cat /tmp/smoke_body)"
grep -q '"chat.completion"' /tmp/smoke_body || fail "unexpected completion body"

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
compose kill replica-1
if ! convergence_status=$(curl -sS -o /tmp/smoke_body -w '%{http_code}' \
  -X POST "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama","messages":[{"role":"user","content":"x"}]}'); then
  fail "request during ejection convergence could not reach the gateway"
fi
case "$convergence_status" in
  200|502) ;;
  *) fail "request during ejection convergence returned $convergence_status" ;;
esac
wait_for_metric 'kairyu_replica_healthy{pool="llama",replica="0"}' "0.0" 10 \
  || fail "replica-1 was not marked unhealthy after it stopped"
for i in $(seq 1 10); do
  [[ "$(chat "user-$i")" == 200 ]] || fail "request after ejection returned non-200"
done

echo "== restart replica-1: prober recovery (gate C2) =="
compose start replica-1
wait_for_metric 'kairyu_replica_healthy{pool="llama",replica="0"}' "1.0" 30 \
  || fail "prober did not restore replica-1"
wait_for "$BASE_URL/readyz" '"ready"' 5 || fail "gateway unready after recovery"

echo "SMOKE PASS"
