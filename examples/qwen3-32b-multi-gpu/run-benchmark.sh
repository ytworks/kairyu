#!/bin/sh
set -eu

cd "$(dirname "$0")"

./run.sh --detach

attempt=0
until curl --fail --silent http://127.0.0.1:8001/readyz >/dev/null 2>&1; do
  attempt=$((attempt + 1))
  if [ "$attempt" -ge 180 ]; then
    echo "Kairyu did not become ready on http://127.0.0.1:8001" >&2
    docker compose logs kairyu >&2
    exit 1
  fi
  sleep 5
done

exec ./benchmark.sh
