#!/usr/bin/env bash
# Runbook §1: FlashInfer backend vs torch oracle on real kernels.
source "$(dirname "$0")/_lib.sh"
preflight --min-gpus 1 --require-module flashinfer --require-executable ninja
pytest_no_skip -m gpu tests/gpu/test_flashinfer_gpu.py -v
run uv run --frozen python scripts/parity_real_model.py
