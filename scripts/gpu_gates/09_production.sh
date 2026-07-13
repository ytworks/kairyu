#!/usr/bin/env bash
# Runbook §9: production bring-up (compose/helm smoke on GPUs).
source "$(dirname "$0")/_lib.sh"
run docker build -f Dockerfile.cuda -t kairyu:cuda .
run docker compose -f deploy/compose/docker-compose.gpu.yaml up -d
run curl -sf http://127.0.0.1:8000/readyz
run uv run python scripts/gpu_gates/check_served_model.py --base-url http://127.0.0.1:8000/v1 --model "$KAIRYU_BENCH_MODEL"
run uv run python bench/serving_bench.py --base-url http://127.0.0.1:8000/v1 --model "$KAIRYU_BENCH_MODEL"
