#!/bin/sh
set -eu

cd "$(dirname "$0")"

# Piping nvidia-smi straight into `wc -l` takes the PIPELINE's status, which is
# `tr`'s success. A broken driver (`nvidia-smi` exits 18 on a driver/library
# version mismatch) then reads as "0 GPUs", and the script blames the GPU count
# for what is actually an unusable driver — a real hour of debugging.
# stderr is NOT merged into the captured list: a warning line carrying a date,
# version or bus id would be counted as another GPU. On failure nvidia-smi is
# re-run so the operator still sees what it actually said.
if ! gpu_list="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)"; then
  echo "nvidia-smi failed; the NVIDIA driver is not usable on this host:" >&2
  nvidia-smi --query-gpu=index --format=csv,noheader >&2 2>&1 || true
  echo "(a driver upgrade without reloading the kernel module reports exactly this)" >&2
  exit 1
fi
# awk, not `grep -c`: grep exits 1 on zero matches and `set -e` would kill the
# script before the found-0 diagnostic below ever runs
unexpected="$(printf '%s\n' "$gpu_list" | awk 'NF && !/^[0-9]+$/ { n++ } END { print n + 0 }')"
if [ "$unexpected" -ne 0 ]; then
  echo "nvidia-smi returned $unexpected line(s) that are not GPU indices:" >&2
  printf '%s\n' "$gpu_list" >&2
  exit 1
fi
gpu_count="$(printf '%s\n' "$gpu_list" | awk 'NF && /^[0-9]+$/ { n++ } END { print n + 0 }')"
if [ "$gpu_count" -ne 8 ]; then
  echo "This DeepSeek-V4-Flash-0731 example uses tensor_parallel_size=8 and requires 8 visible NVIDIA GPUs; found $gpu_count" >&2
  exit 1
fi

echo "Using all 8 visible GPUs (tensor_parallel_size=8, expert parallel)"
# exported so the compose port mapping honours a custom PORT
PORT="${PORT:-8001}"
export PORT
echo "Serving the OpenAI-compatible API on http://127.0.0.1:$PORT/v1"
exec docker compose up --build "$@"
