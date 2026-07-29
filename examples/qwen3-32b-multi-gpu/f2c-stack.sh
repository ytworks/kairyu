#!/bin/sh
set -eu

image="${KAIRYU_F2C_IMAGE:-}"
case "$image" in
  *@sha256:*)
    repository="${image%@sha256:*}"
    digest="${image##*@sha256:}"
    ;;
  *)
    echo "KAIRYU_F2C_IMAGE must be an immutable repository@sha256:<64 hex> reference" >&2
    exit 2
    ;;
esac
if [ -z "$repository" ] || [ "${#digest}" -ne 64 ]; then
  echo "KAIRYU_F2C_IMAGE must be an immutable repository@sha256:<64 hex> reference" >&2
  exit 2
fi
case "$digest" in
  *[!0-9a-f]*)
    echo "KAIRYU_F2C_IMAGE digest must contain exactly 64 lowercase hexadecimal characters" >&2
    exit 2
    ;;
esac

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
compose_file="$script_dir/f2c-compose.yaml"

if [ "$#" -eq 0 ]; then
  echo "usage: KAIRYU_F2C_IMAGE=repository@sha256:... $0 <compose arguments>" >&2
  echo "example: $0 up -d --wait" >&2
  exit 2
fi

exec docker compose -f "$compose_file" "$@"
