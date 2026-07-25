#!/bin/sh
# Run the Fugu quality suite against an already-running Qwen3-32B service and
# print the accuracy report against the published Fugu scores.
#
# The perf harness (benchmark.sh) measures throughput; this measures answers.
set -eu

cd "$(dirname "$0")"

repo_root="$(cd ../.. && pwd)"
port="${PORT:-8001}"
base_url="http://127.0.0.1:${port}/v1"
model="${MODEL:-qwen3-32b}"
judge_model="${JUDGE_MODEL:-$model}"
# A full run is tens of thousands of judged items (HLE alone is 2,500 per
# target), so the example caps items per slot. BENCH_LIMIT=0 runs everything.
bench_limit="${BENCH_LIMIT:-20}"
attempts="${ATTEMPTS:-1}"
concurrency="${BENCH_CONCURRENCY:-8}"
results_dir="${RESULTS_DIR:-$(pwd)/results/fugu}"
run_id="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"

# Qwen/Qwen3-32B is a text-generation causal LM -- the vision family is the
# separate Qwen3-VL. Declaring the target vision-capable would let CharXiv and
# HLE's image rows be "measured" on prompts whose image parts the text-only chat
# template drops, and persist an apparently completed score. VISION=1 for a
# genuinely multimodal deployment.
if [ -n "${VISION:-}" ]; then
  vision_flag=""
  printf '[fugu] target declared vision-capable (VISION set)\n'
else
  vision_flag="--no-vision"
fi

if ! command -v uv >/dev/null 2>&1; then
  cat >&2 <<'MSG'
This script runs `kairyu bench` on the host because the suite needs the dataset
extra (datasets/huggingface_hub/pillow/h5py/tqdm), which the serving image does
not carry. Install uv (https://docs.astral.sh/uv/) and re-run.
MSG
  exit 1
fi

printf '[fugu] waiting for %s\n' "http://127.0.0.1:${port}/readyz"
attempt=0
until curl --fail --silent "http://127.0.0.1:${port}/readyz" >/dev/null 2>&1; do
  attempt=$((attempt + 1))
  if [ "$attempt" -ge 180 ]; then
    echo "Kairyu is not ready on port ${port}; start it with ./run.sh --detach" >&2
    exit 1
  fi
  sleep 5
done

# Preflight the served model by exact id, the same rule gate 09 uses: a healthy
# gateway can pass readyz while serving something else entirely.
served="$(curl --fail --silent "${base_url}/models")" || {
  echo "could not read ${base_url}/models" >&2
  exit 1
}
if ! printf '%s' "$served" | grep -q "\"id\"[[:space:]]*:[[:space:]]*\"${model}\""; then
  printf 'model %s is not served at %s\n' "$model" "$base_url" >&2
  printf '%s\n' "$served" >&2
  exit 1
fi
printf '[fugu] serving %s\n' "$model"

set -- \
  --base-url "$base_url" \
  --model "$model" \
  $vision_flag \
  --judge-base-url "$base_url" \
  --judge-model "$judge_model" \
  --attempts "$attempts" \
  --concurrency "$concurrency" \
  --results-dir "$results_dir" \
  --run-id "$run_id"

if [ "$bench_limit" -gt 0 ]; then
  printf '[fugu] SUBSET RUN: at most %s items per benchmark (BENCH_LIMIT=0 for the full suite)\n' \
    "$bench_limit"
  printf '[fugu] subset and fixture runs are marked in scoreboard.md and comparison.md,\n'
  printf '[fugu] which withhold every delta against the published Fugu scores\n'
  set -- "$@" --limit "$bench_limit"
else
  printf '[fugu] FULL RUN: every item in every slot; this takes hours\n'
fi

# Plumbing check: committed synthetic fixtures, no dataset downloads at all.
# Scores from a fixture run are meaningless and the scoreboard says so.
if [ -n "${OFFLINE_FIXTURES:-}" ]; then
  printf '[fugu] OFFLINE FIXTURES: synthetic stand-in data, scores are not meaningful\n'
  set -- "$@" --offline-fixtures
fi

[ -n "${REASONING_EFFORT:-}" ] && set -- "$@" --reasoning-effort "$REASONING_EFFORT"
[ -n "${EXTRA_BODY:-}" ] && set -- "$@" --extra-body "$EXTRA_BODY"
[ -n "${JUDGE_REASONING_EFFORT:-}" ] &&
  set -- "$@" --judge-reasoning-effort "$JUDGE_REASONING_EFFORT"
[ -n "${BENCH_ONLY:-}" ] && set -- "$@" --only "$BENCH_ONLY"
[ -n "${BENCH_EXCLUDE:-}" ] && set -- "$@" --exclude "$BENCH_EXCLUDE"

printf '[fugu] run id %s -> %s\n' "$run_id" "$results_dir"
cd "$repo_root"
# --extra bench brings the dataset downloaders and tqdm; progress and the
# comparison report are printed by the runner itself.
uv run --extra bench kairyu bench run "$@"

printf '\n[fugu] scoreboard: %s/%s/scoreboard.md\n' "$results_dir" "$run_id"
printf '[fugu] accuracy report: %s/%s/comparison.md\n' "$results_dir" "$run_id"
