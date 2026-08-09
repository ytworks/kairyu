#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
EXAMPLE="$ROOT/examples/deepseek-v4-flash-0731-8gpu"
ROW=long-context:ruler-niah-1024k
TOPOLOGY=${1:-ep4}

case "$TOPOLOGY" in
  ep4|ep8) ;;
  *) echo "usage: $0 [ep4|ep8]" >&2; exit 2 ;;
esac

"$EXAMPLE/bench.sh" list | grep -Fx "$ROW" >/dev/null
export KAIRYU_TOPOLOGY=$TOPOLOGY
exec "$EXAMPLE/bench.sh" kairyu "$ROW"
