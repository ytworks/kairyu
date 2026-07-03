#!/usr/bin/env bash
# Goal G5 gates: fleet elasticity + KV routing under load.
source "$(dirname "$0")/_lib.sh"
run uv run pytest tests/unit/test_fleet_elastic.py tests/unit/test_kv_routing.py -v --no-cov
run bash scripts/kind_smoke.sh
