#!/usr/bin/env bash
# Smoke only the Kairyu service from the Open WebUI topology. The mutable WebUI
# image is neither pulled nor browser-tested here.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$REPO_ROOT/deploy/compose/docker-compose.webui.yaml"
COMPOSE_VALIDATOR="$REPO_ROOT/scripts/validate_compose_binds.py"
BASE_URL="${BASE_URL:-http://localhost:8000}"
EXPECTED_WEBUI_BASE_URL="http://kairyu:8000/v1"
EXPECTED_MODEL_ID="default"
response_file=""

compose() { docker compose -f "$COMPOSE_FILE" "$@"; }
curl_bounded() { curl --connect-timeout 2 --max-time 10 "$@"; }

cleanup() {
  compose down --volumes --remove-orphans >/dev/null 2>&1 || true
  [[ -z "$response_file" ]] || rm -f "$response_file" || true
}

fail() {
  echo "WEBUI SMOKE FAIL: $1" >&2
  compose logs --tail 30 kairyu >&2 || true
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

echo "== validate WebUI compose binds =="
uv run --project "$REPO_ROOT" --no-dev python "$COMPOSE_VALIDATOR" "$COMPOSE_FILE"

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

# Install cleanup only after all pre-start validation succeeds. From this point
# onward every attempted startup is paired with a teardown.
trap cleanup EXIT

echo "== up Kairyu only (no WebUI image pull) =="
compose up -d --build kairyu

echo "== readiness =="
wait_for "$BASE_URL/readyz" '"ready"' 60 || fail "Kairyu never became ready"

echo "== exact model discovery =="
models="$(curl_bounded -sf "$BASE_URL/v1/models")" \
  || fail "/v1/models request failed"
if ! printf '%s' "$models" | python3 -c '
import json
import sys

expected = sys.argv[1]
data = json.load(sys.stdin)
model_ids = [entry.get("id") for entry in data.get("data", [])]
if model_ids != [expected]:
    print(f"expected model ids {[expected]!r}, got {model_ids!r}", file=sys.stderr)
    raise SystemExit(1)
' "$EXPECTED_MODEL_ID"; then
  fail "/v1/models did not expose exactly $EXPECTED_MODEL_ID"
fi

echo "== non-stream completion =="
response_file="$(mktemp "${TMPDIR:-/tmp}/kairyu-webui-smoke.XXXXXX")"
curl_bounded -sf -o "$response_file" -X POST "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"default","messages":[{"role":"user","content":"hello"}],"stream":false}' \
  || fail "completion request failed"
if ! python3 -c '
import json
import sys

expected = sys.argv[1]
data = json.load(sys.stdin)
if data.get("object") != "chat.completion" or data.get("model") != expected:
    raise SystemExit(1)
' "$EXPECTED_MODEL_ID" < "$response_file"; then
  fail "unexpected completion response"
fi

echo "WEBUI SMOKE PASS"
