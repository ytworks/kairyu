#!/usr/bin/env bash
# End-to-end Open WebUI smoke: first-user setup, the admin's Responses filter,
# direct/auto streaming, persistence, gateway outage, and recovery without
# restarting Open WebUI.
#
# The verdict depends only on the code under test: waits observe states or
# events, and their bounds only detect a hang.
#
# Usage: webui_smoke.sh [--setup-only | --no-setup]
#   (none)        build and pull the images, then run the test
#   --setup-only  build and pull the images (the networked setup; CI retries it)
#   --no-setup    run the test on those images; never build or pull
# The two split modes run as separate commands, so they need
# WEBUI_SMOKE_PROJECT_NAME to name one Compose project (and its images).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$REPO_ROOT/deploy/compose/docker-compose.webui.yaml"
COMPOSE_VALIDATOR="$REPO_ROOT/scripts/validate_compose_binds.py"
BASE_URL="${BASE_URL:-http://localhost:8000}"
EXPECTED_WEBUI_BASE_URL="http://kairyu:8000/v1"
# Hang bounds, far above the expected times.
COMPOSE_WAIT_SECONDS="${COMPOSE_WAIT_SECONDS:-600}"
HANG_ATTEMPTS=90
MODE="${1:-}"
case "$MODE" in
  "" | --setup-only | --no-setup) ;;
  *) echo "usage: $0 [--setup-only | --no-setup]" >&2; exit 2 ;;
esac
if [[ -n "$MODE" && -z "${WEBUI_SMOKE_PROJECT_NAME:-}" ]]; then
  echo "$MODE needs WEBUI_SMOKE_PROJECT_NAME so setup and test share one project" >&2
  exit 2
fi
SMOKE_PROJECT_NAME="${WEBUI_SMOKE_PROJECT_NAME:-kairyu-webui-smoke-$$-${RANDOM}}"
# The demo's admin credentials (README) would make Open WebUI create that admin
# and close sign-up, so the browser's fresh first-user signup could not run.
# Empty values also override a deploy/compose/.env file.
export WEBUI_ADMIN_EMAIL="" WEBUI_ADMIN_PASSWORD=""

# A dedicated project gives every run its own network, containers, and data
# volume. Cleanup must never remove a developer's normal Compose demo data.
compose() {
  docker compose --project-name "$SMOKE_PROJECT_NAME" -f "$COMPOSE_FILE" "$@"
}
# Bounds only detect a hung connection; they are far above any response time.
curl_bounded() { curl --connect-timeout 30 --max-time 120 "$@"; }

cleanup() {
  compose --profile browser-smoke down --volumes --remove-orphans \
    >/dev/null 2>&1 || true
}

fail() {
  echo "WEBUI E2E FAIL: $1" >&2
  compose logs --tail 80 kairyu webui >&2 || true
  exit 1
}

hang() {
  echo "WEBUI E2E HANG: $1 (a hang bound, not a speed limit)" >&2
  compose logs --tail 80 kairyu webui >&2 || true
  exit 1
}

wait_for() { # wait_for <url> <substring> <attempts>
  local url=$1 want=$2 attempts=$3 body
  for _ in $(seq 1 "$attempts"); do
    body=$(curl_bounded -sf "$url" 2>/dev/null) \
      && [[ "$body" == *"$want"* ]] \
      && return 0
    sleep 2
  done
  return 1
}

browser_phase() {
  local phase=$1
  compose --profile browser-smoke run --rm --no-deps --pull never \
    -e WEBUI_SMOKE_PHASE="$phase" browser
}

assert_embedding_contract() {
  local body
  body=$(curl_bounded -sf \
    -H "Content-Type: application/json" \
    -d '{"model":"embed-small","input":["semantic vector retrieval","fresh strawberries"],"encoding_format":"float"}' \
    "$BASE_URL/v1/embeddings") \
    || fail "Kairyu production embedding request failed"
  printf '%s' "$body" | python3 -c '
import json
import math
import sys

payload = json.load(sys.stdin)
assert payload["model"] == "embed-small", payload
assert payload["usage"] == {"prompt_tokens": 10, "total_tokens": 10}, payload
vectors = [item["embedding"] for item in payload["data"]]
assert len(vectors) == 2, payload
assert all(len(vector) == 384 for vector in vectors), payload
assert all(all(math.isfinite(value) for value in vector) for vector in vectors), payload
assert all(math.isclose(math.sqrt(sum(value * value for value in vector)), 1.0, rel_tol=1e-4, abs_tol=1e-4) for vector in vectors), payload
assert vectors[0] != vectors[1], payload
' || fail "Kairyu embedding dimensions/order/usage contract failed"
}

# await_metrics <failure> <check> [args...]: the counters only grow, so a check
# that exits 3 (below its minimum) is polled until it passes. A request the
# browser saw finish may be counted just after its response reached the browser.
await_metrics() {
  local failure=$1 body status
  shift
  for _ in $(seq 1 "$HANG_ATTEMPTS"); do
    body=$(curl_bounded -sf "$BASE_URL/metrics") \
      || fail "Kairyu metrics were unavailable: $failure"
    status=0
    printf '%s' "$body" | "$@" || status=$?
    case "$status" in
      0) return 0 ;;
      3) sleep 1 ;;
      *) fail "$failure" ;;
    esac
  done
  hang "$failure"
}

assert_embedding_metrics() {
  local minimum=$1 phase=$2
  await_metrics "RAG embedding metrics did not prove $phase traffic" python3 -c '
import sys

minimum = float(sys.argv[1])
value = None
for line in sys.stdin:
    if line.startswith("kairyu_requests_total{") and "code=\"200\"" in line and "model=\"embed-small\"" in line:
        value = float(line.rsplit(" ", 1)[1])
        break
if value is None or value < minimum:
    print(
        f"embed-small successful request count {value!r} is below required {minimum:g}",
        file=sys.stderr,
    )
    raise SystemExit(3)
' "$minimum"
}

setup_images() {
  echo "== setup: build and pull images =="
  compose --profile browser-smoke build kairyu browser
  compose pull webui
}

if [[ "$MODE" == --setup-only ]]; then
  setup_images
  exit 0
fi

echo "== validate WebUI compose binds =="
uv run --frozen --project "$REPO_ROOT" --no-dev \
  python "$COMPOSE_VALIDATOR" "$COMPOSE_FILE"

echo "== validate rendered WebUI endpoint contract =="
rendered_config="$(compose --profile browser-smoke config --format json)"
printf '%s' "$rendered_config" | python3 -c '
import json
import sys

expected = sys.argv[1]
data = json.load(sys.stdin)
environment = data.get("services", {}).get("webui", {}).get("environment", {})
actual = environment.get("OPENAI_API_BASE_URL")
if actual != expected:
    print(
        f"WEBUI CONFIG FAIL: expected OPENAI_API_BASE_URL={expected!r}, got {actual!r}",
        file=sys.stderr,
    )
    raise SystemExit(1)
api_type = json.loads(environment.get("OPENAI_API_CONFIGS") or "{}").get("0", {}).get("api_type")
if api_type != "responses":
    print(
        f"WEBUI CONFIG FAIL: connection 0 must use api_type=responses, got {api_type!r}",
        file=sys.stderr,
    )
    raise SystemExit(1)
' "$EXPECTED_WEBUI_BASE_URL"

# The fresh browser account is Open WebUI's admin; webui-setup signs in as it
# to install the Responses filter.
browser_account() { # browser_account <WEBUI_SMOKE_EMAIL|WEBUI_SMOKE_PASSWORD>
  printf '%s' "$rendered_config" | python3 -c '
import json
import sys

print(json.load(sys.stdin)["services"]["browser"]["environment"][sys.argv[1]])
' "$1"
}
smoke_email="$(browser_account WEBUI_SMOKE_EMAIL)"
smoke_password="$(browser_account WEBUI_SMOKE_PASSWORD)"

if [[ -z "$MODE" ]]; then
  setup_images
fi

# From the first startup attempt onward, always remove containers and the
# ephemeral first-user database so every run starts from the same state.
trap cleanup EXIT

assert_responses_wire() {
  local minimum=$1 phase=$2
  await_metrics "Open WebUI did not use the Responses API during $phase" python3 -c '
import sys

minimum = float(sys.argv[1])
counts = {}
for line in sys.stdin:
    if line.startswith("kairyu_request_duration_seconds_count{"):
        labels, value = line.rsplit(" ", 1)
        counts[labels] = float(value)
responses = counts.get("kairyu_request_duration_seconds_count{path=\"/v1/responses\"}", 0.0)
chat = counts.get("kairyu_request_duration_seconds_count{path=\"/v1/chat/completions\"}", 0.0)
if chat:
    raise SystemExit(f"Open WebUI still reached /v1/chat/completions {chat:g} time(s)")
if responses < minimum:
    print(f"/v1/responses request count {responses:g} is below required {minimum:g}", file=sys.stderr)
    raise SystemExit(3)
' "$minimum"
}

echo "== start Kairyu and Open WebUI =="
compose up -d --no-build --pull never --wait --wait-timeout "$COMPOSE_WAIT_SECONDS" kairyu webui \
  || fail "Kairyu/Open WebUI topology did not become healthy (unhealthy, exited, or past the ${COMPOSE_WAIT_SECONDS}s hang bound)"

echo "== production embedding dimensions, order, normalization, and exact usage =="
assert_embedding_contract

echo "== fresh first-user signup =="
browser_phase signup || fail "fresh first-user signup failed"

echo "== admin installs the Responses filter =="
compose run --rm --no-deps --pull never \
  -e WEBUI_ADMIN_EMAIL="$smoke_email" \
  -e WEBUI_ADMIN_PASSWORD="$smoke_password" \
  webui-setup || fail "Open WebUI Responses filter setup failed"

echo "== direct/auto streaming and persistence =="
browser_phase initial || fail "initial browser contract failed"
assert_embedding_metrics 4 "initial ingest/retrieval/chat"
assert_responses_wire 5 "initial RAG/streaming/UI chat"

echo "== gateway outage is visible in the existing Open WebUI instance =="
compose stop kairyu || fail "could not stop Kairyu"
browser_phase outage || fail "browser did not surface the gateway outage"

echo "== gateway recovery works without restarting Open WebUI =="
compose start kairyu || fail "could not restart Kairyu"
wait_for "$BASE_URL/readyz" '"ready"' "$HANG_ATTEMPTS" \
  || hang "Kairyu did not become ready after restart"
browser_phase recovery || fail "browser did not recover after Kairyu restart"
assert_embedding_metrics 2 "post-restart retrieval/chat"
assert_responses_wire 3 "post-restart RAG/streaming/UI chat"

echo "WEBUI E2E PASS"
