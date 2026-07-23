#!/bin/sh
set -eu

cd "$(dirname "$0")"

progress_interval_s="${PROGRESS_INTERVAL_S:-5}"

printf '[startup] starting Qwen3-32B service\n'
./run.sh --detach

printf '[startup] waiting for readiness at http://127.0.0.1:8001/readyz\n'
attempt=0
elapsed_s=0
until curl --fail --silent http://127.0.0.1:8001/readyz >/dev/null 2>&1; do
  attempt=$((attempt + 1))
  elapsed_s=$((elapsed_s + progress_interval_s))
  printf '[startup] waiting for readiness (elapsed %ss)\n' "$elapsed_s"
  if [ "$attempt" -ge 180 ]; then
    echo "Kairyu did not become ready on http://127.0.0.1:8001" >&2
    docker compose logs kairyu >&2
    exit 1
  fi
  sleep "$progress_interval_s"
done

printf '[startup] ready after %ss\n' "$elapsed_s"
printf '[benchmark] starting\n'
exec ./benchmark.sh
