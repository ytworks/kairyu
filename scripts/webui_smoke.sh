#!/usr/bin/env bash
# End-to-end Open WebUI smoke: first-user setup, direct/auto streaming,
# persistence, gateway outage, and recovery without restarting Open WebUI.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$REPO_ROOT/deploy/compose/docker-compose.webui.yaml"
COMPOSE_VALIDATOR="$REPO_ROOT/scripts/validate_compose_binds.py"
BASE_URL="${BASE_URL:-http://localhost:8000}"
EXPECTED_WEBUI_BASE_URL="http://kairyu:8000/v1"
COMPOSE_WAIT_SECONDS="${COMPOSE_WAIT_SECONDS:-240}"
SMOKE_PROJECT_NAME="${WEBUI_SMOKE_PROJECT_NAME:-kairyu-webui-smoke-$$-${RANDOM}}"

# A dedicated project gives every run its own network, containers, and data
# volume. Cleanup must never remove a developer's normal Compose demo data.
compose() {
  docker compose --project-name "$SMOKE_PROJECT_NAME" -f "$COMPOSE_FILE" "$@"
}
curl_bounded() { curl --connect-timeout 2 --max-time 10 "$@"; }

cleanup() {
  compose --profile browser-smoke down --volumes --remove-orphans \
    >/dev/null 2>&1 || true
}

fail() {
  echo "WEBUI E2E FAIL: $1" >&2
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
  compose --profile browser-smoke run --rm --no-deps \
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

assert_embedding_metrics() {
  local minimum=$1 phase=$2 body
  body=$(curl_bounded -sf "$BASE_URL/metrics") \
    || fail "Kairyu metrics were unavailable during $phase"
  printf '%s' "$body" | python3 -c '
import sys

minimum = float(sys.argv[1])
value = None
for line in sys.stdin:
    if line.startswith("kairyu_requests_total{") and "code=\"200\"" in line and "model=\"embed-small\"" in line:
        value = float(line.rsplit(" ", 1)[1])
        break
if value is None or value < minimum:
    raise SystemExit(
        f"embed-small successful request count {value!r} is below required {minimum:g}"
    )
' "$minimum" || fail "RAG embedding metrics did not prove $phase traffic"
}

echo "== validate WebUI compose binds =="
uv run --frozen --project "$REPO_ROOT" --no-dev \
  python "$COMPOSE_VALIDATOR" "$COMPOSE_FILE"

echo "== validate rendered WebUI endpoint contract =="
rendered_config="$(compose config --format json)"
printf '%s' "$rendered_config" | python3 -c '
import json
import sys

expected = sys.argv[1]
data = json.load(sys.stdin)
actual = data.get("services", {}).get("webui", {}).get("environment", {}).get(
    "OPENAI_API_BASE_URL"
)
if actual != expected:
    print(
        f"WEBUI CONFIG FAIL: expected OPENAI_API_BASE_URL={expected!r}, got {actual!r}",
        file=sys.stderr,
    )
    raise SystemExit(1)
' "$EXPECTED_WEBUI_BASE_URL"

# From the first startup attempt onward, always remove containers and the
# ephemeral first-user database so every run starts from the same state.
trap cleanup EXIT

echo "== start Kairyu and Open WebUI =="
compose up -d --build --wait --wait-timeout "$COMPOSE_WAIT_SECONDS" kairyu webui \
  || fail "Kairyu/Open WebUI topology did not become healthy"

echo "== build pinned Playwright runner =="
compose --profile browser-smoke build browser \
  || fail "Playwright runner build failed"

echo "== production embedding dimensions, order, normalization, and exact usage =="
assert_embedding_contract

echo "== fresh-user direct/auto streaming and persistence =="
browser_phase initial || fail "initial browser contract failed"
assert_embedding_metrics 4 "initial ingest/retrieval/chat"

echo "== gateway outage is visible in the existing Open WebUI instance =="
compose stop kairyu || fail "could not stop Kairyu"
browser_phase outage || fail "browser did not surface the gateway outage"

echo "== gateway recovery works without restarting Open WebUI =="
compose start kairyu || fail "could not restart Kairyu"
wait_for "$BASE_URL/readyz" '"ready"' 90 \
  || fail "Kairyu did not become ready after restart"
browser_phase recovery || fail "browser did not recover after Kairyu restart"
assert_embedding_metrics 2 "post-restart retrieval/chat"

echo "WEBUI E2E PASS"
