#!/usr/bin/env bash
# Run one networked CI setup command (dependency sync, downloads, image pulls
# and builds) with bounded retries. Setup is never part of a test verdict: a
# command that still fails after the last attempt fails its setup step, which
# is reported as a setup failure naming the command.
set -euo pipefail

ATTEMPTS=3
if (($# == 0)); then
  echo "usage: $0 <command> [args...]" >&2
  exit 2
fi
for attempt in $(seq 1 "$ATTEMPTS"); do
  if "$@"; then
    exit 0
  fi
  echo "setup attempt ${attempt}/${ATTEMPTS} failed: $*" >&2
  if ((attempt < ATTEMPTS)); then
    sleep $((attempt * 15))
  fi
done
echo "SETUP FAILED after ${ATTEMPTS} attempts (not a test verdict): $*" >&2
exit 1
