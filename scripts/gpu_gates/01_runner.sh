#!/usr/bin/env bash
# Runbook §1: FlashInfer backend vs torch oracle on real kernels.
source "$(dirname "$0")/_lib.sh"
run uv run pytest -m gpu tests/gpu/test_flashinfer_gpu.py -v
run uv run python scripts/parity_real_model.py
