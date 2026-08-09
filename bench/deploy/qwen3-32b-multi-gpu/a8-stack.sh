#!/bin/sh
set -eu

usage() {
  echo "usage: KAIRYU_A8_IMAGE=<image> KAIRYU_A8_IMAGE_ID=sha256:<64 hex> $0 <run-dir> <compose arguments>" >&2
  echo "example: $0 bench/results/g2-a8-run up -d --wait" >&2
  exit 2
}

image="${KAIRYU_A8_IMAGE:-}"
case "$image" in
  *[![:space:]]*) ;;
  *)
    echo "KAIRYU_A8_IMAGE must be nonblank" >&2
    exit 2
    ;;
esac
case "$image" in
  *[[:space:]]*)
    echo "KAIRYU_A8_IMAGE must be one Docker image reference without whitespace" >&2
    exit 2
    ;;
esac

image_id="${KAIRYU_A8_IMAGE_ID:-}"
case "$image_id" in
  sha256:*) digest="${image_id#sha256:}" ;;
  *)
    echo "KAIRYU_A8_IMAGE_ID must be sha256:<64 lowercase hex>" >&2
    exit 2
    ;;
esac
if [ "${#digest}" -ne 64 ]; then
  echo "KAIRYU_A8_IMAGE_ID must be sha256:<64 lowercase hex>" >&2
  exit 2
fi
case "$digest" in
  *[!0-9a-f]*)
    echo "KAIRYU_A8_IMAGE_ID must be sha256:<64 lowercase hex>" >&2
    exit 2
    ;;
esac

if [ "$#" -lt 2 ]; then
  usage
fi
run_dir_arg=$1
shift
case "$run_dir_arg" in
  *[![:space:]]*) ;;
  *)
    echo "run-dir must be nonblank" >&2
    exit 2
    ;;
esac

# Do not let pass-through arguments replace this stack's Compose file or
# project identity.  In particular, an accidental `-p other down` must never
# target containers owned by another project.
is_up=false
force_recreate=false
for compose_arg in "$@"; do
  case "$compose_arg" in
    -p|--project-name|-p?*|--project-name=*|-f|--file|-f?*|--file=*|\
    --project-directory|--project-directory=*|--env-file|--env-file=*)
      echo "compose arguments must not override the A8 project or file scope: $compose_arg" >&2
      exit 2
      ;;
    up) is_up=true ;;
    --force-recreate) force_recreate=true ;;
  esac
done
if [ "$is_up" = true ] && [ "$force_recreate" != true ]; then
  echo "A8 up requires --force-recreate so source and startup config are current" >&2
  exit 2
fi

# Resolve one caller-owned evidence directory without cleaning or replacing it.
# `--` and quoting keep a dash-prefixed or whitespace-containing path literal.
mkdir -p -- "$run_dir_arg"
if [ ! -d "$run_dir_arg" ] || [ ! -w "$run_dir_arg" ]; then
  echo "run-dir must resolve to a writable directory: $run_dir_arg" >&2
  exit 2
fi
run_dir=$(CDPATH= cd -- "$run_dir_arg" && pwd -P)
if [ "$run_dir" = "/" ]; then
  echo "run-dir must not resolve to the filesystem root" >&2
  exit 2
fi
placement_log="$run_dir/placements.jsonl"
if [ -L "$placement_log" ] || { [ -e "$placement_log" ] && [ ! -f "$placement_log" ]; }; then
  echo "placements.jsonl must be absent or an existing regular file" >&2
  exit 2
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
compose_file="$script_dir/a8-compose.yaml"
export KAIRYU_A8_RUN_DIR="$run_dir"

# A fixed project and fixed Compose file scope every lifecycle operation to A8.
# The external model volume is never created, removed, or renamed by this script.
exec docker compose \
  --project-name kairyu-qwen3-32b-a8 \
  -f "$compose_file" \
  "$@"
