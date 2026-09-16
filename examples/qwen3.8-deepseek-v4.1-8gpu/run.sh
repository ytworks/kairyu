#!/usr/bin/env bash
exec python3 "$(dirname "$0")/control.py" "${1:-up}"
