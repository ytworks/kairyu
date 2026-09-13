#!/usr/bin/env bash
# Run the example's measured verifications with the repository interpreter.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
NVME_ROOT="${NVME_STORAGE_ROOT:-/mnt/nvme/kairyu}"
mkdir -p "$NVME_ROOT/bench-tmp"
export TMPDIR="$NVME_ROOT/bench-tmp"
export PATH="$REPO_ROOT/.venv/bin:$PATH"
exec "$REPO_ROOT/.venv/bin/python" "$SCRIPT_DIR/verification.py" "$@"
