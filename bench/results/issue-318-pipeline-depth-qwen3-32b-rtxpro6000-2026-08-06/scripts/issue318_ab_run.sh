#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <main|candidate> <run-index>" >&2
  exit 2
fi

arm=$1
run_index=$2
case "$arm" in
  main)
    source_root=/tmp/kairyu-issue318-main-baseline-20260806
    ;;
  candidate)
    source_root=/home/y-takagi/kairyu
    ;;
  *)
    echo "invalid arm: $arm" >&2
    exit 2
    ;;
esac
if [[ ! "$run_index" =~ ^[0-9]+$ ]]; then
  echo "invalid run index: $run_index" >&2
  exit 2
fi

container_name="issue318-refined-${arm}-r${run_index}"
artifact="/tmp/issue318-refined-${arm}-r${run_index}.json"
config=/home/y-takagi/kairyu/bench/results/issue-333-proc-http-qwen3-32b-rtxpro6000-2026-08-05/configs/00-issue333-tp4-sharegpt-r0-kairyu.yaml
trace=/tmp/a6-traces/g2-a6-traces.json
image=kairyu-qwen3-32b-kairyu:latest

if docker inspect "$container_name" >/dev/null 2>&1; then
  echo "container already exists: $container_name" >&2
  exit 1
fi
if [[ -e "$artifact" ]]; then
  echo "artifact already exists: $artifact" >&2
  exit 1
fi

cleanup() {
  if docker inspect "$container_name" >/dev/null 2>&1; then
    docker stop --time 20 "$container_name" >/dev/null || true
  fi
}
trap cleanup EXIT

echo "starting ${arm} run ${run_index}"
docker run --rm -d \
  --name "$container_name" \
  --gpus '"device=0,1,2,3"' \
  --ipc=host \
  --ulimit memlock=-1 \
  -p 127.0.0.1:8002:8000 \
  -v kairyu-qwen3-32b_qwen3-32b:/models:ro \
  -v "$source_root":/workspace:ro \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$source_root/kairyu":/app/kairyu:ro \
  -v "$config":/etc/kairyu/config.yaml:ro \
  -e KAIRYU_ATTENTION_BACKEND=flashinfer \
  --entrypoint /app/.venv/bin/kairyu \
  "$image" serve /etc/kairyu/config.yaml

ready=0
for _ in $(seq 1 120); do
  if curl -sf http://127.0.0.1:8002/readyz >/dev/null; then
    ready=1
    break
  fi
  if ! docker inspect "$container_name" >/dev/null 2>&1; then
    echo "container exited before readiness" >&2
    exit 1
  fi
  sleep 2
done
if [[ "$ready" -ne 1 ]]; then
  echo "server readiness timed out" >&2
  docker logs --tail 200 "$container_name" >&2
  exit 1
fi

echo "measuring ${arm} run ${run_index}"
(
  cd "$source_root"
  PYTHONDONTWRITEBYTECODE=1 /home/y-takagi/kairyu/.venv/bin/python \
    /tmp/issue318_measure.py \
    --endpoint http://127.0.0.1:8002 \
    --trace-bundle "$trace" \
    --label "refined-${arm}-r${run_index}" \
    --source "$source_root" \
    --config "$config" \
    --container-name "$container_name" \
    --out "$artifact"
)
echo "completed ${arm} run ${run_index}: ${artifact}"
