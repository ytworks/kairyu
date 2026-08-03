#!/bin/sh
# Start (or reuse) Qwen3-32B, provision every benchmark dependency, run the
# Fugu quality suite, and print the accuracy report against the published
# Fugu scores.  The only required operator input is HF_TOKEN after accepting
# the gated dataset licenses on Hugging Face.
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
reasoning_effort="${REASONING_EFFORT:-high}"
judge_reasoning_effort="${JUDGE_REASONING_EFFORT:-low}"
extra_body_default='{"chat_template_kwargs":{"enable_thinking":true}}'
extra_body="${EXTRA_BODY:-$extra_body_default}"
# Official tau-three v1.0.1 still ships from the tau2-bench repository and
# exposes the `tau2` CLI. Pin the resolved tag commit: a moving Git ref would
# make two nominally identical benchmark runs use different tasks or grading.
tau_commit='fc0055dc4e0a316c3f83133267fbd6faaa770992'
tau_repository='https://github.com/sierra-research/tau2-bench.git'
tau_requirement="tau2[knowledge] @ git+${tau_repository}@${tau_commit}"
exec_image_tag="${KAIRYU_BENCH_EXEC_TAG:-kairyu-bench-exec:local}"

# Validated before anything slow happens. `[ "$x" -gt 0 ]` returns status 2 on a
# non-integer, and inside an `if` condition that does NOT trip `set -e`, so an
# unvalidated BENCH_LIMIT=typo would fall through to the else branch and launch a
# FULL RUN -- hours and tens of thousands of judged items, which is exactly what
# the default cap exists to avoid.
require_non_negative_integer() {
  case "$2" in
    '' | *[!0-9]*)
      printf '%s must be a non-negative integer, got %s\n' "$1" "$2" >&2
      exit 2
      ;;
  esac
}

require_non_negative_integer BENCH_LIMIT "$bench_limit"
require_non_negative_integer ATTEMPTS "$attempts"
require_non_negative_integer BENCH_CONCURRENCY "$concurrency"
require_non_negative_integer PORT "$port"

if [ -z "${HF_TOKEN:-}" ]; then
  cat >&2 <<'MSG'
HF_TOKEN is required for the gated GPQA Diamond, HLE, and LiveCodeBench Pro
datasets. Accept their Hugging Face licenses, export it in the host environment,
then run:

  export HF_TOKEN=hf_...
  ./examples/qwen3-32b-multi-gpu/fugu-benchmark.sh
MSG
  exit 2
fi
# Read only the environment inherited from the host shell. Export explicitly so
# both the Compose model downloader and the host-side `uv run` benchmark inherit
# the same credential; no .env file or token argument is read by this script.
export HF_TOKEN

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

# A non-editable Git install contains the tau2 Python package but not its task
# data. Keep one commit-addressed source checkout and point the official
# TAU2_DATA_DIR contract at it. An explicit TAU2_DATA_DIR is accepted for
# operators who already manage that checkout themselves.
if [ -z "${TAU2_DATA_DIR:-}" ]; then
  if ! command -v git >/dev/null 2>&1; then
    echo "git is required to provision the pinned tau-three task data" >&2
    exit 1
  fi
  bench_cache="${KAIRYU_BENCH_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/kairyu/benchmarks}"
  tau_parent="$bench_cache/harnesses"
  tau_checkout="$tau_parent/tau2-$tau_commit"
  mkdir -p "$tau_parent"
  if [ ! -d "$tau_checkout/.git" ]; then
    if [ -e "$tau_checkout" ]; then
      printf 'incomplete tau-three checkout exists at %s; move it aside and retry\n' \
        "$tau_checkout" >&2
      exit 1
    fi
    tau_tmp="$(mktemp -d "$tau_parent/.tau2.XXXXXX")"
    cleanup_tau_tmp() {
      case "${tau_tmp:-}" in
        "$tau_parent"/.tau2.*) rm -rf -- "$tau_tmp" ;;
      esac
    }
    trap cleanup_tau_tmp 0 1 2 15
    printf '[setup] downloading tau-three v1.0.1 task data\n'
    git clone --filter=blob:none --no-checkout "$tau_repository" "$tau_tmp/repo"
    git -C "$tau_tmp/repo" checkout --detach "$tau_commit"
    mv "$tau_tmp/repo" "$tau_checkout"
    rmdir "$tau_tmp"
    tau_tmp=""
    trap - 0 1 2 15
  fi
  actual_tau_commit="$(git -C "$tau_checkout" rev-parse HEAD)"
  if [ "$actual_tau_commit" != "$tau_commit" ]; then
    printf 'tau-three checkout commit mismatch: expected %s, got %s\n' \
      "$tau_commit" "$actual_tau_commit" >&2
    exit 1
  fi
  TAU2_DATA_DIR="$tau_checkout/data"
  export TAU2_DATA_DIR
fi
if [ ! -d "$TAU2_DATA_DIR/tau2/domains/banking_knowledge" ]; then
  printf 'TAU2_DATA_DIR lacks banking_knowledge task data: %s\n' "$TAU2_DATA_DIR" >&2
  exit 1
fi
printf '[setup] tau-three data %s\n' "$TAU2_DATA_DIR"

if ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
  echo "Docker is required for agentic benchmarks and sandboxed code execution" >&2
  exit 1
fi

# Generated answers from LiveCodeBench and SciCode are untrusted code. Resolve
# the configured local tag to an immutable image ID, building the hash-pinned
# runtime when it is absent. A caller may instead name an existing local image
# or repository digest through KAIRYU_BENCH_EXEC_IMAGE.
exec_image_source="${KAIRYU_BENCH_EXEC_IMAGE:-$exec_image_tag}"
if [ -z "${KAIRYU_BENCH_EXEC_IMAGE:-}" ] &&
  ! docker image inspect "$exec_image_source" >/dev/null 2>&1; then
  printf '[setup] building benchmark execution image %s\n' "$exec_image_tag"
  docker build \
    -f "$repo_root/deploy/bench/Dockerfile.exec" \
    -t "$exec_image_tag" \
    "$repo_root/deploy/bench"
fi
exec_image="$(docker image inspect --format '{{.Id}}' "$exec_image_source")" || {
  printf 'benchmark execution image %s is unavailable\n' "$exec_image_source" >&2
  exit 1
}
case "$exec_image" in
  sha256:????????????????????????????????????????????????????????????????) ;;
  *)
    printf 'Docker returned a non-immutable execution image id: %s\n' "$exec_image" >&2
    exit 1
    ;;
esac
printf '[setup] sandbox image %s\n' "$exec_image"

# fugu-benchmark.sh is itself the one-command entry point. Reuse a healthy
# service, otherwise let the sibling hardware-checked launcher download the
# model, build the serving image, and start Compose on the selected port.
if curl --fail --silent "http://127.0.0.1:${port}/readyz" >/dev/null 2>&1; then
  printf '[startup] Kairyu already ready on port %s\n' "$port"
else
  printf '[startup] starting Qwen3-32B service\n'
  HF_TOKEN="$HF_TOKEN" PORT="$port" ./run.sh --detach
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
# The ids are extracted with a regex over the RESPONSE, then compared as
# strings: interpolating MODEL into the pattern would make the "exact id" check
# a glob, so MODEL=qwen3.32b would accept a deployment serving qwen3X32b.
served_ids="$(
  printf '%s' "$served" |
    grep -o '"id"[[:space:]]*:[[:space:]]*"[^"]*"' |
    sed 's/.*"\([^"]*\)"$/\1/'
)"
found=0
while IFS= read -r served_id; do
  if [ "$served_id" = "$model" ]; then
    found=1
    break
  fi
done <<SERVED_IDS
$served_ids
SERVED_IDS
if [ "$found" -ne 1 ]; then
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
  --reasoning-effort "$reasoning_effort" \
  --judge-reasoning-effort "$judge_reasoning_effort" \
  --extra-body "$extra_body" \
  --exec-runner docker \
  --exec-image "$exec_image" \
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

[ -n "${BENCH_ONLY:-}" ] && set -- "$@" --only "$BENCH_ONLY"
[ -n "${BENCH_EXCLUDE:-}" ] && set -- "$@" --exclude "$BENCH_EXCLUDE"

printf '[fugu] run id %s -> %s\n' "$run_id" "$results_dir"
cd "$repo_root"
# `bench` owns datasets/reporting, `bench-agentic` owns SWE-Bench Pro and
# Harbor, and the pinned --with requirement owns official tau-three (whose
# package and CLI remain named tau2). uv provisions them automatically.
uv run --frozen \
  --extra bench \
  --extra bench-agentic \
  --with "$tau_requirement" \
  kairyu bench run "$@"

printf '\n[fugu] scoreboard: %s/%s/scoreboard.md\n' "$results_dir" "$run_id"
printf '[fugu] accuracy report: %s/%s/comparison.md\n' "$results_dir" "$run_id"
