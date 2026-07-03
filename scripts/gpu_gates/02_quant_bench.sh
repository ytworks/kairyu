#!/usr/bin/env bash
# Runbook §2: quant kernels vs CPU oracles + first serving benches.
source "$(dirname "$0")/_lib.sh"
run uv run pytest -m gpu tests/gpu/test_quant_kernels.py -v
run uv run python bench/serving_bench.py --base-url http://127.0.0.1:8000/v1 --model default
