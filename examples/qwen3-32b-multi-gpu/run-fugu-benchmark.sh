#!/bin/sh
# One command: start Qwen3-32B on every visible GPU, wait for readiness, run the
# Fugu quality suite, and print the accuracy report against the published Fugu
# scores.
set -eu

cd "$(dirname "$0")"

port="${PORT:-8001}"
progress_interval_s="${PROGRESS_INTERVAL_S:-5}"
# run.sh exports PORT into the compose port mapping, so a custom port really is
# where the service listens rather than where this script waits in vain.
export PORT="$port"

if curl --fail --silent "http://127.0.0.1:${port}/readyz" >/dev/null 2>&1; then
  printf '[startup] Kairyu already ready on port %s\n' "$port"
else
  printf '[startup] starting Qwen3-32B service\n'
  ./run.sh --detach

  printf '[startup] waiting for readiness at http://127.0.0.1:%s/readyz\n' "$port"
  attempt=0
  elapsed_s=0
  until curl --fail --silent "http://127.0.0.1:${port}/readyz" >/dev/null 2>&1; do
    attempt=$((attempt + 1))
    elapsed_s=$((elapsed_s + progress_interval_s))
    printf '[startup] waiting for readiness (elapsed %ss)\n' "$elapsed_s"
    if [ "$attempt" -ge 180 ]; then
      echo "Kairyu did not become ready on http://127.0.0.1:${port}" >&2
      docker compose logs kairyu >&2
      exit 1
    fi
    sleep "$progress_interval_s"
  done
  printf '[startup] ready after %ss\n' "$elapsed_s"
fi

printf '[fugu] starting quality suite\n'
exec ./fugu-benchmark.sh
