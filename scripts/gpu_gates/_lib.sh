#!/usr/bin/env bash
# Shared helpers for the gpu_gates scripts (m19 D3). Every script supports
# --dry-run: print the exact commands without executing anything.
set -euo pipefail
KAIRYU_BENCH_MODEL=${KAIRYU_BENCH_MODEL:-default}
DRY_RUN=0
for arg in "$@"; do [ "$arg" = "--dry-run" ] && DRY_RUN=1; done
run() {
  echo "+ $*"
  if [ "$DRY_RUN" -eq 0 ]; then "$@"; fi
}
preflight() {
  run uv run --frozen python scripts/test_prerequisites.py "$@"
}
pytest_no_skip() {
  run uv run --frozen pytest --fail-on-skip "$@"
}
