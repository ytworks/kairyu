#!/bin/sh
set -eu
here=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
venv_bin="$here/../../.venv/bin"
PATH="$venv_bin:$PATH"
export PATH
exec "$venv_bin/python" "$here/benchmark.py" "$@"
