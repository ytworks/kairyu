#!/bin/sh
set -eu
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "usage: ./run.sh <vllm|kairyu> [up|down|status|logs]" >&2
  exit 2
fi
backend="$1"
action="${2:-up}"
case "$backend" in vllm|kairyu) ;; *) echo "unknown backend: $backend" >&2; exit 2 ;; esac
case "$action" in up|down|status|logs) ;; *) echo "unknown action: $action" >&2; exit 2 ;; esac
exec python3 "$(dirname "$0")/../_shared/examplectl.py" "$(dirname "$0")" "$backend" "$action"
