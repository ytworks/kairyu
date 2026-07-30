#!/usr/bin/env bash
# Goal G5 gates: fleet elasticity + KV routing under load.
source "$(dirname "$0")/_lib.sh"
preflight --require-executable docker --require-executable kind \
  --require-executable kubectl --require-executable helm \
  --require-module zmq
run bash scripts/helm_integration.sh
pytest_no_skip -m "not helm" \
  tests/unit/test_fleet_elastic.py tests/unit/test_kv_routing.py -v --no-cov
run bash scripts/kind_smoke.sh
