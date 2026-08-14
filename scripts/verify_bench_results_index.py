#!/usr/bin/env python3
"""Verify the retained benchmark-results catalog against Git-tracked evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from evidence.legacy_index import ResultsIndexError, validate_results_repository


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (default: parent of scripts/)",
    )
    args = parser.parse_args(argv)
    try:
        index = validate_results_repository(args.repo)
    except ResultsIndexError as error:
        parser.exit(1, f"{error}\n")
    print(
        f"verified {len(index.artifacts)} Git-tracked top-level benchmark "
        "result artifacts"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
