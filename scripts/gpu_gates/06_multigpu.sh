#!/usr/bin/env bash
# Runbook §6 / G2 A-gates: the m16 dist suite with backend=nccl.
source "$(dirname "$0")/_lib.sh"
run uv run pytest tests/dist -v
export KAIRYU_DIST_BACKEND=nccl
run uv run pytest tests/dist -v
run uv run python bench/serving_bench.py --base-url http://127.0.0.1:8000/v1 --model default
