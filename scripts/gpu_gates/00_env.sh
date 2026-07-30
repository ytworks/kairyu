#!/usr/bin/env bash
# Runbook §0: environment audit + env record.
source "$(dirname "$0")/_lib.sh"
run uv sync --frozen --extra gpu --extra hf --extra fleet
preflight --min-gpus 1 --require-module flashinfer --require-executable ninja
run nvidia-smi
run uv run --frozen python scripts/gpu_gates/record_env.py
