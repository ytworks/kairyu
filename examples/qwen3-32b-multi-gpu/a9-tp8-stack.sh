#!/bin/sh
set -eu

usage() {
  echo "usage: KAIRYU_A9_IMAGE=<image> KAIRYU_A9_IMAGE_ID=sha256:<64 hex> $0 <run-dir> <compose arguments>" >&2
  echo "example: $0 bench/results/g2-a9-tp8-run up -d --force-recreate --wait" >&2
  exit 2
}

image="${KAIRYU_A9_IMAGE:-}"
case "$image" in
  *[![:space:]]*) ;;
  *)
    echo "KAIRYU_A9_IMAGE must be nonblank" >&2
    exit 2
    ;;
esac
case "$image" in
  *[[:space:]]*)
    echo "KAIRYU_A9_IMAGE must be one Docker image reference without whitespace" >&2
    exit 2
    ;;
esac

image_id="${KAIRYU_A9_IMAGE_ID:-}"
case "$image_id" in
  sha256:*) digest="${image_id#sha256:}" ;;
  *)
    echo "KAIRYU_A9_IMAGE_ID must be sha256:<64 lowercase hex>" >&2
    exit 2
    ;;
esac
if [ "${#digest}" -ne 64 ]; then
  echo "KAIRYU_A9_IMAGE_ID must be sha256:<64 lowercase hex>" >&2
  exit 2
fi
case "$digest" in
  *[!0-9a-f]*)
    echo "KAIRYU_A9_IMAGE_ID must be sha256:<64 lowercase hex>" >&2
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

# Every operation is locked to this stack's project and Compose file.  In
# particular, a pass-through `-p` or `-f` must never target another stack.
is_up=false
force_recreate=false
for compose_arg in "$@"; do
  case "$compose_arg" in
    -p|--project-name|-p?*|--project-name=*|-f|--file|-f?*|--file=*|\
    --project-directory|--project-directory=*|--env-file|--env-file=*)
      echo "compose arguments must not override the A9 TP8 project or file scope: $compose_arg" >&2
      exit 2
      ;;
    up) is_up=true ;;
    --force-recreate) force_recreate=true ;;
  esac
done
if [ "$is_up" = true ] && [ "$force_recreate" != true ]; then
  echo "A9 TP8 up requires --force-recreate so source and startup config are current" >&2
  exit 2
fi

# Preserve a caller-owned evidence directory; never clean or replace it.
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
compose_file="$script_dir/a9-tp8-compose.yaml"
repo_root=$(CDPATH= cd -- "$script_dir/../.." && pwd -P)
a8_source_commit=86d49223ffcdba6052428474bf0d9094c6791fed
a8_source_archive_sha256=18ff0eabe0dd63862e00230d93cc8edb18c52c97f9544b7ded3e8088e5e3d66d
a8_runtime_files_sha256=2e695979ef6cf4eb9e5708549af82f8ee6776a49765f1f1408a598ceac893417
run_parent=$(dirname -- "$run_dir")
run_name=$(basename -- "$run_dir")
runtime_parent="$run_parent/.$run_name.a8-source-$a8_source_commit"
runtime_dir="$runtime_parent/kairyu"
source_archive="$runtime_parent/source.tar"

if [ "$is_up" = true ]; then
  if [ ! -e "$runtime_parent" ]; then
    mkdir -p -- "$runtime_parent"
    git -C "$repo_root" archive \
      --format=tar \
      --output="$source_archive" \
      "$a8_source_commit" \
      kairyu
    tar -xf "$source_archive" -C "$runtime_parent"
  fi
  if [ -L "$runtime_parent" ] ||
     [ ! -d "$runtime_parent" ] ||
     [ -L "$runtime_dir" ] ||
     [ ! -d "$runtime_dir" ] ||
     [ -L "$source_archive" ] ||
     [ ! -f "$source_archive" ]; then
    echo "pinned A8 runtime source is incomplete; use a fresh run-dir" >&2
    exit 2
  fi
  observed_archive_sha256=$(sha256sum "$source_archive" | awk '{print $1}')
  if [ "$observed_archive_sha256" != "$a8_source_archive_sha256" ]; then
    echo "pinned A8 source archive SHA-256 differs; use a fresh run-dir" >&2
    exit 2
  fi
  observed_runtime_sha256=$(
    python3 -c '
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
files = []
for path in sorted(root.rglob("*")):
    if path.is_file():
        payload = path.read_bytes()
        files.append(
            {
                "path": f"kairyu/{path.relative_to(root).as_posix()}",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
encoded = json.dumps(
    files,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
).encode()
print(hashlib.sha256(encoded).hexdigest())
' "$runtime_dir"
  )
  if [ "$observed_runtime_sha256" != "$a8_runtime_files_sha256" ]; then
    echo "materialized A8 runtime file tree differs; use a fresh run-dir" >&2
    exit 2
  fi
fi

export KAIRYU_A9_RUN_DIR="$run_dir"
export KAIRYU_A9_RUNTIME_DIR="$runtime_dir"

# The external checkpoint volume is never created, removed, or renamed here.
exec docker compose \
  --project-name kairyu-qwen3-32b-a9-tp8 \
  -f "$compose_file" \
  "$@"
