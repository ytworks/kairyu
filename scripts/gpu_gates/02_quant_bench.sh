#!/usr/bin/env bash
# Runbook §2: quant kernels vs CPU oracles + first serving benches.
source "$(dirname "$0")/_lib.sh"
preflight --min-gpus 1 --min-compute-capability 10.0 \
  --require-module flashinfer --require-module triton
pytest_no_skip -m gpu tests/gpu/test_quant_kernels.py -v
run uv run --frozen python verification/l1/performance/serving_bench.py --base-url http://127.0.0.1:8000/v1 --model default
