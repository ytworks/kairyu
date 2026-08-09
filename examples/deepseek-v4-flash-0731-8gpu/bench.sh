#!/bin/sh
set -eu
exec "$(dirname "$0")/../../.venv/bin/python" "$(dirname "$0")/../_shared/benchctl.py" "$(dirname "$0")" "$@"
