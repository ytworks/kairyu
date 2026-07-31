#!/usr/bin/env bash
# GPU-only Open WebUI image-chat gate through Kairyu and stock vLLM TP8.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BASE_COMPOSE_FILE="$REPO_ROOT/deploy/compose/docker-compose.webui.yaml"
VLM_COMPOSE_FILE="$REPO_ROOT/deploy/compose/docker-compose.webui-vlm.yaml"
COMPOSE_VALIDATOR="$REPO_ROOT/scripts/validate_compose_binds.py"
FIXTURE_GENERATOR="$REPO_ROOT/scripts/make_vlm_image_fixtures.py"
API_SMOKE="$REPO_ROOT/scripts/webui_vlm_api_smoke.py"
BROWSER_SMOKE="$REPO_ROOT/scripts/webui_vlm_browser_smoke.mjs"

KAIRYU_VLM_BASE_URL="${KAIRYU_VLM_BASE_URL:-http://127.0.0.1:8000/v1}"
VLM_MODEL="${KAIRYU_VLM_MODEL:-qwen3-vl-32b}"
VLM_MAX_PROMPT_TOKENS="${KAIRYU_VLM_MAX_PROMPT_TOKENS:-8192}"
VLM_REQUEST_TIMEOUT_SECONDS="${KAIRYU_VLM_REQUEST_TIMEOUT_SECONDS:-300}"
COMPOSE_WAIT_SECONDS="${KAIRYU_VLM_COMPOSE_WAIT_SECONDS:-2400}"
SMOKE_PROJECT_NAME="${WEBUI_VLM_SMOKE_PROJECT_NAME:-kairyu-webui-vlm-smoke}"
KEEP_STACK="${WEBUI_VLM_KEEP_STACK:-0}"
FIXTURE_DIR="$(mktemp -d /tmp/kairyu-vlm-fixtures.XXXXXX)"
STARTED=0

compose() {
  docker compose --project-name "$SMOKE_PROJECT_NAME" \
    -f "$BASE_COMPOSE_FILE" -f "$VLM_COMPOSE_FILE" "$@"
}

cleanup() {
  if [[ "$STARTED" == "1" && "$KEEP_STACK" != "1" ]]; then
    # Preserve the project-scoped model cache across runs. A 32B checkpoint
    # must not be downloaded again merely because one smoke run completed.
    compose down --remove-orphans >/dev/null 2>&1 || true
  fi
  rm -rf -- "$FIXTURE_DIR"
}

fail() {
  echo "WEBUI VLM FAIL: $1" >&2
  if [[ "$STARTED" == "1" ]]; then
    compose logs --tail 160 vllm kairyu webui >&2 || true
  fi
  exit 1
}

trap cleanup EXIT

echo "== validate GPU-only Compose binds =="
uv run --frozen --project "$REPO_ROOT" --no-dev \
  python "$COMPOSE_VALIDATOR" "$BASE_COMPOSE_FILE" "$VLM_COMPOSE_FILE" \
  || fail "Compose bind validation failed"

echo "== generate deterministic RED/BLUE PNG fixtures =="
python3 "$FIXTURE_GENERATOR" "$FIXTURE_DIR" \
  || fail "PNG fixture generation failed"

echo "== start pinned Qwen3-VL-32B TP8, Kairyu, and OpenWebUI =="
STARTED=1
compose up -d --build --wait --wait-timeout "$COMPOSE_WAIT_SECONDS" \
  vllm kairyu webui \
  || fail "VLM topology did not become healthy within ${COMPOSE_WAIT_SECONDS}s"

echo "== direct Kairyu image semantics, exact usage, stream, and URL safety =="
python3 "$API_SMOKE" \
  --base-url "$KAIRYU_VLM_BASE_URL" \
  --model "$VLM_MODEL" \
  --red "$FIXTURE_DIR/red.png" \
  --blue "$FIXTURE_DIR/blue.png" \
  --max-prompt-tokens "$VLM_MAX_PROMPT_TOKENS" \
  --timeout "$VLM_REQUEST_TIMEOUT_SECONDS" \
  || fail "direct Kairyu VLM API contract failed"

echo "== build pinned Playwright runner =="
compose --profile browser-smoke build browser \
  || fail "Playwright runner build failed"

echo "== real OpenWebUI file-input image chat =="
compose --profile browser-smoke run --rm --no-deps \
  -v "$FIXTURE_DIR:/fixtures:ro" \
  -v "$BROWSER_SMOKE:/work/webui_vlm_browser_smoke.mjs:ro" \
  -e EXPECTED_VLM_MODEL="$VLM_MODEL" \
  -e WEBUI_VLM_FIXTURE=/fixtures/red.png \
  -e WEBUI_VLM_RESPONSE_TIMEOUT_MS="$((VLM_REQUEST_TIMEOUT_SECONDS * 1000))" \
  browser node /work/webui_vlm_browser_smoke.mjs \
  || fail "OpenWebUI browser image-chat contract failed"

echo "WEBUI VLM PASS"
