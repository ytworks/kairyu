#!/usr/bin/env bash
# Runbook §3: remaining GPU-deferred seams (graphs, streams).
source "$(dirname "$0")/_lib.sh"
run uv run pytest -m gpu tests/gpu -v
