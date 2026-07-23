#!/bin/sh
set -eu

cd "$(dirname "$0")"

repo_root="$(cd ../.. && pwd)"
num_requests="${NUM_REQUESTS:-128}"
concurrency="${CONCURRENCY:-32}"
max_tokens="${MAX_TOKENS:-128}"
ttft_slo_s="${TTFT_SLO_S:-2.0}"
timeout_s="${TIMEOUT_S:-600}"
progress_interval_s="${PROGRESS_INTERVAL_S:-5}"
metrics_url="http://127.0.0.1:8001/metrics"

metric_request_count() {
  model="$1"
  metrics="$(curl --fail --silent "$metrics_url" 2>/dev/null)" || return 1
  printf '%s\n' "$metrics" |
    awk -v model="$model" '
      $1 ~ /^kairyu_requests_total\{/ &&
      index($1, "model=\"" model "\"") {
        if ($2 !~ /^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$/) {
          malformed = 1
          next
        }
        total += $2
      }
      END {
        if (malformed) exit 1
        printf "%.0f\n", total
      }
    '
}

monitor_progress() {
  baseline="$1"
  elapsed_s=0
  while :; do
    sleep "$progress_interval_s"
    elapsed_s=$((elapsed_s + progress_interval_s))
    current=""
    if [ -n "$baseline" ]; then
      current="$(metric_request_count qwen3-32b)" || current=""
    fi
    if [ -n "$current" ]; then
      completed=$((current - baseline))
      [ "$completed" -ge 0 ] || completed=0
      [ "$completed" -le "$num_requests" ] || completed="$num_requests"
      printf '[benchmark] completed %s/%s (elapsed %ss)\n' \
        "$completed" "$num_requests" "$elapsed_s"
    else
      printf '[benchmark] running (elapsed %ss)\n' "$elapsed_s"
    fi
  done
}

stop_progress_monitor() {
  if [ -n "${progress_pid:-}" ]; then
    kill "$progress_pid" 2>/dev/null || true
    wait "$progress_pid" 2>/dev/null || true
    progress_pid=""
  fi
}

curl --fail --silent --show-error http://127.0.0.1:8001/readyz >/dev/null

gpu_count="$(
  docker compose exec -T kairyu sh -ec \
    "nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d '[:space:]'"
)"
image_id="$(docker compose images -q kairyu)"
if [ -z "$image_id" ]; then
  echo "Kairyu image not found; start the service first" >&2
  exit 1
fi

printf '[benchmark] requests=%s concurrency=%s max_tokens=%s GPUs/TP=%s\n' \
  "$num_requests" "$concurrency" "$max_tokens" "$gpu_count"

baseline="$(metric_request_count qwen3-32b)" || baseline=""
progress_pid=""
trap stop_progress_monitor EXIT HUP INT TERM
monitor_progress "$baseline" &
progress_pid=$!

benchmark_status=0
docker run --rm --network host \
  --entrypoint python \
  --volume "$repo_root/bench:/bench:ro" \
  --volume "$(pwd)/results:/results" \
  "$image_id" \
  /bench/serving_bench.py \
  --base-url http://127.0.0.1:8001 \
  --model qwen3-32b \
  --num-requests "$num_requests" \
  --concurrency "$concurrency" \
  --max-tokens "$max_tokens" \
  --ttft-slo-s "$ttft_slo_s" \
  --timeout "$timeout_s" \
  --tensor-parallel "$gpu_count" \
  --results-dir /results || benchmark_status=$?

stop_progress_monitor
trap - EXIT HUP INT TERM
if [ "$benchmark_status" -ne 0 ]; then
  exit "$benchmark_status"
fi

printf '[report] generating Markdown report\n'
docker run --rm \
  --entrypoint python \
  --volume "$(pwd)/benchmark_report.py:/opt/kairyu/benchmark_report.py:ro" \
  --volume "$(pwd)/results:/results" \
  "$image_id" \
  /opt/kairyu/benchmark_report.py \
  /results \
  --output /results/report.md

printf '\nReport: %s/results/report.md\n' "$(pwd)"
