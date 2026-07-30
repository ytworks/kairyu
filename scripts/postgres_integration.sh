#!/usr/bin/env bash
# Locally reproducible PostgreSQL integration suite.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
POSTGRES_IMAGE=${KAIRYU_TEST_POSTGRES_IMAGE:-postgres:17.6-bookworm@sha256:f3bd19c606e442c3d7bdfa8002e03fe260a1023351e0ea4598032022b68dd6e3}
CONTAINER="kairyu-postgres-test-$$"
PASSWORD="kairyu-test"
DATABASE="kairyu_test"

cleanup() {
  status=$?
  trap - EXIT
  if [[ "$status" -ne 0 ]]; then
    docker logs "$CONTAINER" >&2 2>/dev/null || true
  fi
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT

cd "$REPO_ROOT"
uv run --frozen python scripts/test_prerequisites.py \
  --require-executable docker \
  --require-module psycopg

docker run --detach --rm \
  --name "$CONTAINER" \
  --publish 127.0.0.1::5432 \
  --env "POSTGRES_PASSWORD=$PASSWORD" \
  --env "POSTGRES_DB=$DATABASE" \
  "$POSTGRES_IMAGE" >/dev/null

for _ in $(seq 1 60); do
  if docker exec "$CONTAINER" pg_isready \
    --username postgres --dbname "$DATABASE" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
docker exec "$CONTAINER" pg_isready \
  --username postgres --dbname "$DATABASE" >/dev/null

binding=$(docker port "$CONTAINER" 5432/tcp | head -n 1)
port=${binding##*:}
[[ "$port" =~ ^[1-9][0-9]*$ ]]
export KAIRYU_TEST_POSTGRES_DSN="postgresql://postgres:${PASSWORD}@127.0.0.1:${port}/${DATABASE}"

uv run --frozen python -c \
  'import os, psycopg; connection = psycopg.connect(os.environ["KAIRYU_TEST_POSTGRES_DSN"]); assert connection.execute("SELECT 1").fetchone() == (1,); connection.close()'
uv run --frozen pytest --fail-on-skip \
  -m postgres tests/unit/test_postgres_batch_store.py -v --no-cov
