#!/bin/sh
set -eu
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "usage: ./run.sh <vllm|kairyu> [up|down|status|logs]" >&2
  exit 2
fi
exec python3 "$(dirname "$0")/../_shared/examplectl.py" "$(dirname "$0")" "$1" "${2:-up}"
