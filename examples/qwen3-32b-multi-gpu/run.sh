#!/bin/sh
set -eu

cd "$(dirname "$0")"

# Piping nvidia-smi straight into `wc -l` takes the PIPELINE's status, which is
# `tr`'s success. A broken driver (`nvidia-smi` exits 18 on a driver/library
# version mismatch) then reads as "0 GPUs", and the script blames the GPU count
# for what is actually an unusable driver — a real hour of debugging.
if ! gpu_list="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>&1)"; then
  echo "nvidia-smi failed; the NVIDIA driver is not usable on this host:" >&2
  echo "$gpu_list" >&2
  echo "(a driver upgrade without reloading the kernel module reports exactly this)" >&2
  exit 1
fi
gpu_count="$(printf '%s\n' "$gpu_list" | grep -c '[0-9]')"
case "$gpu_count" in
  2|4|8) ;;
  *)
    echo "Qwen3-32B requires 2, 4, or 8 visible NVIDIA GPUs; found $gpu_count" >&2
    exit 1
    ;;
esac

echo "Using all $gpu_count visible GPUs"
# exported so the compose port mapping honours a custom PORT
PORT="${PORT:-8001}"
export PORT
echo "Serving the OpenAI-compatible API on http://127.0.0.1:$PORT/v1"
exec docker compose up --build "$@"
