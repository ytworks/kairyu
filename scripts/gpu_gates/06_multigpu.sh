#!/usr/bin/env bash
# Runbook §6 / G2 A-gates: gloo parity plus explicit NCCL coverage.
source "$(dirname "$0")/_lib.sh"
KAIRYU_QWEN3_32B_MODEL=${KAIRYU_QWEN3_32B_MODEL:-/models/qwen3-32b}
run uv run pytest tests/dist -v
run uv run pytest -m gpu tests/gpu/test_tp_sampling_owner_nccl.py -v
run uv run python bench/tp_sampling_owner_bench.py --world-size 8 --rounds 8 --trials-per-round 32 --output /tmp/kairyu-tp-sampling-owner-gate.json --assert-gate
run uv run python bench/tp_sampling_owner_bench.py --verify /tmp/kairyu-tp-sampling-owner-gate.json --assert-gate
run uv run python bench/tp_sampling_owner_qwen.py --model-path "$KAIRYU_QWEN3_32B_MODEL" --output /tmp/kairyu-tp-sampling-owner-qwen-gate.json --assert-gate
run uv run python bench/tp_sampling_owner_qwen.py --verify /tmp/kairyu-tp-sampling-owner-qwen-gate.json --assert-gate
run uv run pytest -m gpu tests/gpu/test_moe_parallel_nccl.py -v
run uv run python bench/serving_bench.py --base-url http://127.0.0.1:8000/v1 --model default
