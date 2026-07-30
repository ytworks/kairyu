#!/usr/bin/env bash
# Locally reproducible Helm rendering and schema suite.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

uv run --frozen python scripts/test_prerequisites.py \
  --require-executable helm \
  --require-module yaml
uv run --frozen pytest --fail-on-skip \
  -m helm tests/unit/test_fleet_elastic.py -q --no-cov
