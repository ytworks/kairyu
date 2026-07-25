#!/bin/sh
set -eu

cd "$(dirname "$0")"

gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d '[:space:]')"
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
