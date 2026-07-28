#!/usr/bin/env bash
set -euo pipefail

kairyu_base_url="${KAIRYU_BASE_URL:-http://127.0.0.1:8000/v1}"
kairyu_model="${KAIRYU_MODEL:-default}"
kairyu_smoke_mode="${KAIRYU_SMOKE_MODE:-tool}"
export KAIRYU_CODEX_API_KEY="${KAIRYU_API_KEY:-local}"

case "$kairyu_smoke_mode" in
  tool)
    smoke_prompt='Run exactly one shell command: pwd. After receiving its output, reply with exactly PASS.'
    ;;
  text)
    smoke_prompt='Reply with exactly PASS. Do not call tools.'
    ;;
  *)
    echo "KAIRYU_SMOKE_MODE must be 'tool' or 'text'" >&2
    exit 2
    ;;
esac

last_message="$(mktemp "${TMPDIR:-/tmp}/kairyu-codex-message.XXXXXX")"
event_log="$(mktemp "${TMPDIR:-/tmp}/kairyu-codex-events.XXXXXX")"
trap 'rm -f -- "$last_message" "$event_log"' EXIT

codex exec \
  --ignore-user-config \
  --ignore-rules \
  --ephemeral \
  --skip-git-repo-check \
  -c 'model_provider="kairyu"' \
  -c "model_providers.kairyu={name=\"Kairyu\",base_url=\"$kairyu_base_url\",env_key=\"KAIRYU_CODEX_API_KEY\",wire_api=\"responses\",stream_max_retries=0}" \
  -c 'model_supports_reasoning_summaries=false' \
  -c 'model_reasoning_effort="minimal"' \
  -c 'web_search="disabled"' \
  -m "$kairyu_model" \
  --sandbox read-only \
  --json \
  --output-last-message "$last_message" \
  "$smoke_prompt" | tee "$event_log"

grep -q 'PASS' "$last_message"
if [[ "$kairyu_smoke_mode" == "tool" ]]; then
  grep -q '"type":"command_execution"' "$event_log"
  grep -q '"status":"completed"' "$event_log"
fi
