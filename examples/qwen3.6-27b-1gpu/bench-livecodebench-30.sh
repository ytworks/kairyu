#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
    echo "usage: ./bench-livecodebench-30.sh <vllm|kairyu|compare>" >&2
    exit 2
fi

case "$1" in
    vllm|kairyu|compare) ;;
    *)
        echo "usage: ./bench-livecodebench-30.sh <vllm|kairyu|compare>" >&2
        exit 2
        ;;
esac

exec "$(dirname "$0")/bench.sh" "$1" accuracy:livecodebench --limit 30 --seed 0
