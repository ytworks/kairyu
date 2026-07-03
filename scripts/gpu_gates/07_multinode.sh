#!/usr/bin/env bash
# Runbook §7 / G2 B-gates: 2-node KV transport over the real fabric.
source "$(dirname "$0")/_lib.sh"
run uv run pytest tests/dist/test_pd_two_process.py -v
run uv run python bench/kv_transfer_bench.py
