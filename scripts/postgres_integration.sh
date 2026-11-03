#!/usr/bin/env bash
# Locally reproducible PostgreSQL integration suite.
#
# Usage: postgres_integration.sh [--setup-only | --no-setup]
#   (none)        pull the pinned image when it is absent, then run the suite
#   --setup-only  pull the pinned image (the networked setup; CI retries it)
#   --no-setup    run the suite on the pulled image; never pull
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# The linux/amd64 manifest of postgres:17.6-bookworm (index sha256:f3bd19c6...):
# a single-platform digest names the same image under every Docker image store.
POSTGRES_IMAGE=${KAIRYU_TEST_POSTGRES_IMAGE:-postgres:17.6-bookworm@sha256:45cd22f8d32e189d245403954882f88e7a8714301fda80dab6da90f1265b25a3}
CONTAINER="kairyu-postgres-test-$$"
PASSWORD="kairyu-test"
DATABASE="kairyu_test"
# Readiness polls the server's state; the bound only detects a hang.
READY_ATTEMPTS=${KAIRYU_TEST_POSTGRES_READY_ATTEMPTS:-180}
MODE=${1:-}
case "$MODE" in
  "" | --setup-only | --no-setup) ;;
  *) echo "usage: $0 [--setup-only | --no-setup]" >&2; exit 2 ;;
esac

if [[ "$MODE" == --setup-only ]]; then
  docker pull "$POSTGRES_IMAGE"
  exit 0
fi
if [[ "$MODE" != --no-setup ]] \
  && ! docker image inspect "$POSTGRES_IMAGE" >/dev/null 2>&1; then
  docker pull "$POSTGRES_IMAGE"
fi

if command -v uv >/dev/null 2>&1; then
  PYTHON_RUN=(uv run --frozen)
elif command -v poetry >/dev/null 2>&1; then
  PYTHON_RUN=(poetry run)
else
  echo "PostgreSQL integration requires uv or poetry" >&2
  exit 2
fi

if [[ ! "$READY_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "KAIRYU_TEST_POSTGRES_READY_ATTEMPTS must be a positive integer" >&2
  exit 2
fi

cleanup() {
  status=$?
  trap - EXIT
  if [[ "$status" -ne 0 ]]; then
    docker logs "$CONTAINER" >&2 || true
  fi
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT

cd "$REPO_ROOT"
"${PYTHON_RUN[@]}" python scripts/test_prerequisites.py \
  --require-executable docker \
  --require-module psycopg

docker run --detach --pull never \
  --name "$CONTAINER" \
  --publish 127.0.0.1::5432 \
  --env "POSTGRES_PASSWORD=$PASSWORD" \
  --env "POSTGRES_DB=$DATABASE" \
  "$POSTGRES_IMAGE" >/dev/null

binding=$(docker port "$CONTAINER" 5432/tcp | head -n 1)
port=${binding##*:}
[[ "$port" =~ ^[1-9][0-9]*$ ]]
export KAIRYU_TEST_POSTGRES_DSN="postgresql://postgres:${PASSWORD}@127.0.0.1:${port}/${DATABASE}"

postgres_ready() {
  "${PYTHON_RUN[@]}" python -c \
    'import os, psycopg; connection = psycopg.connect(os.environ["KAIRYU_TEST_POSTGRES_DSN"], connect_timeout=2); assert connection.execute("SELECT 1").fetchone() == (1,); connection.close()' \
    >/dev/null 2>&1
}

# The official image briefly runs an initialization-only Unix-socket server
# before stopping it and starting the final TCP server. In-container
# pg_isready can observe that temporary process and race the shutdown. Probe
# the exact published DSN that the tests will use instead.
ready=false
for _ in $(seq 1 "$READY_ATTEMPTS"); do
  if postgres_ready; then
    ready=true
    break
  fi
  sleep 1
done
if [[ "$ready" != true ]]; then
  echo "PostgreSQL HANG: no external TCP query succeeded after $READY_ATTEMPTS attempts (a hang bound, not a speed limit)" >&2
  exit 1
fi

"${PYTHON_RUN[@]}" pytest --fail-on-skip \
  -m postgres \
  tests/unit/test_postgres_batch_store.py \
  tests/unit/test_postgres_node_model_prestage.py \
  tests/unit/test_postgres_runner_leadership.py \
  tests/unit/test_postgres_runner_scaling_log.py \
  tests/unit/test_postgres_runner_startup_admission.py \
  tests/unit/test_postgres_request_store.py \
  -v --no-cov
