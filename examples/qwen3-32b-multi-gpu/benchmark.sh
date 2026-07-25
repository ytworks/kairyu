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

# `sh -ec "... | wc -l"` returns wc's status, so a failed nvidia-smi inside the
# container would be reported as a GPU/TP count of 0 in the benchmark header —
# a wrong number recorded next to real measurements
if ! gpu_list="$(
  docker compose exec -T kairyu sh -ec \
    "nvidia-smi --query-gpu=index --format=csv,noheader" 2>/dev/null
)"; then
  echo "nvidia-smi failed inside the kairyu container:" >&2
  docker compose exec -T kairyu sh -ec \
    "nvidia-smi --query-gpu=index --format=csv,noheader" >&2 2>&1 || true
  exit 1
fi
# a wrong GPU/TP count would be recorded in the header next to real measurements
unexpected="$(printf '%s\n' "$gpu_list" | awk 'NF && !/^[0-9]+$/ { n++ } END { print n + 0 }')"
if [ "$unexpected" -ne 0 ]; then
  echo "nvidia-smi returned $unexpected line(s) that are not GPU indices:" >&2
  printf '%s\n' "$gpu_list" >&2
  exit 1
fi
gpu_count="$(printf '%s\n' "$gpu_list" | awk 'NF && /^[0-9]+$/ { n++ } END { print n + 0 }')"
# run.sh and the compose startup both gate on this; without it here an empty GPU
# list parses cleanly to 0 and is then printed in the benchmark header AND passed
# as `--tensor-parallel 0` — the wrong-number-next-to-real-measurements case this
# whole change exists to prevent
case "$gpu_count" in
  2|4|8) ;;
  *)
    echo "Qwen3-32B requires 2, 4, or 8 visible NVIDIA GPUs; found $gpu_count" >&2
    printf '%s\n' "$gpu_list" >&2
    exit 1
    ;;
esac
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
