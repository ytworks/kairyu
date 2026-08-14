#!/usr/bin/env python3
"""Smoke every supported checkout-only verification CLI invocation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from verification.registry import load_registry, validate_repository


@dataclass(frozen=True, slots=True)
class SmokeResult:
    label: str
    returncode: int
    stdout: str
    stderr: str


def _run(command: list[str], *, label: str, root: Path) -> SmokeResult:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as error:
        return SmokeResult(
            label=label,
            returncode=124,
            stdout=error.stdout or "",
            stderr=f"timed out after {error.timeout} seconds",
        )
    return SmokeResult(
        label=label,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def verify(root: Path, *, workers: int) -> None:
    root = root.resolve()
    errors = validate_repository(root)
    if errors:
        raise RuntimeError("repository verification check failed:\n" + "\n".join(errors))

    jobs: list[tuple[list[str], str]] = []
    for entry in load_registry():
        if "path" in entry.invocations:
            jobs.append(
                (
                    [sys.executable, entry.path, "--help"],
                    f"python {entry.path} --help",
                )
            )
        if "module" in entry.invocations:
            jobs.append(
                (
                    [sys.executable, "-m", entry.module, "--help"],
                    f"python -m {entry.module} --help",
                )
            )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_run, command, label=label, root=root)
            for command, label in jobs
        ]
        results = [future.result() for future in futures]

    failed = [
        result
        for result in results
        if result.returncode != 0 or "usage:" not in result.stdout.lower()
    ]
    if failed:
        details = []
        for result in failed:
            details.append(
                f"{result.label}: exit={result.returncode}\n"
                f"stdout:\n{result.stdout[-1000:]}\n"
                f"stderr:\n{result.stderr[-1000:]}"
            )
        raise RuntimeError("verification CLI smoke failed:\n" + "\n\n".join(details))
    print(
        f"verified {len(load_registry())} verification gates "
        f"through {len(results)} declared --help invocations"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (default: parent of scripts/)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="parallel subprocesses (default: 4)",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    verify(args.repo, workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
