#!/bin/sh
set -eu
exec "$(dirname "$0")/../../.venv/bin/python" "$(dirname "$0")/verification.py" "$@"
