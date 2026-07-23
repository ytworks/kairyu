#!/bin/sh
set -eu

cd "$(dirname "$0")"

num_requests="${NUM_REQUESTS:-128}"
concurrency="${CONCURRENCY:-32}"
max_tokens="${MAX_TOKENS:-128}"
ttft_slo_s="${TTFT_SLO_S:-2.0}"
timeout_s="${TIMEOUT_S:-600}"

docker compose exec -T kairyu \
  curl --fail --silent --show-error http://127.0.0.1:8000/readyz >/dev/null

gpu_count="$(
  docker compose exec -T kairyu sh -ec \
    "nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d '[:space:]'"
)"

docker compose exec -T kairyu python /bench/serving_bench.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen3-32b \
  --num-requests "$num_requests" \
  --concurrency "$concurrency" \
  --max-tokens "$max_tokens" \
  --ttft-slo-s "$ttft_slo_s" \
  --timeout "$timeout_s" \
  --tensor-parallel "$gpu_count" \
  --results-dir /results

docker compose exec -T kairyu python /opt/kairyu/benchmark_report.py \
  /results \
  --output /results/report.md

printf '\nReport: %s/results/report.md\n' "$(pwd)"
