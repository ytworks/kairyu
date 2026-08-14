#!/usr/bin/env bash
# Runbook §7 local prerequisite: two processes plus TCP-loopback KV transfer.
# This script does not claim 2-node/NCCL/RDMA fabric evidence.
source "$(dirname "$0")/_lib.sh"
preflight --require-module transformers
pytest_no_skip tests/dist/test_pd_two_process.py -v
run uv run --frozen python verification/l1/performance/kv_transfer_bench.py
