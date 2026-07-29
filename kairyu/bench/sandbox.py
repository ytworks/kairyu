"""Backward-compatible local execution helpers.

The pluggable runner contract lives in :mod:`kairyu.bench.execution`.  These
functions retain the original public API for trusted development and direct
tests; suite adapters receive their configured runner through ``RunContext``.
"""

from __future__ import annotations

from collections.abc import Mapping

from kairyu.bench.execution import ExecResult, LocalExecutionRunner

_LOCAL_RUNNER = LocalExecutionRunner()


def run_python(
    code: str,
    *,
    stdin: str = "",
    timeout_s: float = 30.0,
    memory_mb: int = 4096,
    files: Mapping[str, bytes] | None = None,
) -> ExecResult:
    return _LOCAL_RUNNER.run_python(
        code,
        stdin=stdin,
        timeout_s=timeout_s,
        memory_mb=memory_mb,
        files=files,
    )


def has_module(name: str) -> bool:
    """Can the sandbox interpreter import `name`? (numpy for SciCode, etc.)"""
    return _LOCAL_RUNNER.has_module(name)
