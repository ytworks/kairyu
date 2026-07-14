#!/usr/bin/env bash
# Shared helpers for the gpu_gates scripts (m19 D3). Every script supports
# --dry-run: print the exact commands (referencing REAL files/tests — pinned
# by tests/unit/test_gpu_gates_scripts.py) without executing anything.
set -euo pipefail
KAIRYU_BENCH_MODEL=${KAIRYU_BENCH_MODEL:-default}
DRY_RUN=0
for arg in "$@"; do [ "$arg" = "--dry-run" ] && DRY_RUN=1; done
run() {
  echo "+ $*"
  if [ "$DRY_RUN" -eq 0 ]; then "$@"; fi
}
