"""Subprocess sandbox for execution-scored benchmarks (LiveCodeBench, SciCode).

Isolation = fresh temp cwd, scrubbed env, `python -I`, and rlimits
(address space / CPU / processes / file size) + wall-clock kill. This is a
guard against runaway benchmark code, NOT a security boundary against a
hostile model — documented in docs/benchmarks.md; a container runner is a
future hook.

Memory rlimits are best-effort: macOS rejects RLIMIT_AS / RLIMIT_DATA with
EINVAL, so there the wall-clock kill is the only containment. Linux (CI and
deploy) enforces the full set.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

_OUTPUT_CAP = 64_000


@dataclass(frozen=True)
class ExecResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def _make_preexec(memory_mb: int, cpu_s: int):
    import resource

    def preexec() -> None:  # pragma: no cover - runs in the forked child
        limit = memory_mb * 1024 * 1024
        for memory_rlimit in (resource.RLIMIT_AS, resource.RLIMIT_DATA):
            try:
                resource.setrlimit(memory_rlimit, (limit, limit))
                break
            except (ValueError, OSError):
                continue  # macOS rejects address-space rlimits with EINVAL
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_FSIZE, (32_000_000, 32_000_000))
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (256, 256))
        except (ValueError, OSError):
            pass  # containers may already sit above a lowerable cap

    return preexec


def run_python(
    code: str,
    *,
    stdin: str = "",
    timeout_s: float = 30.0,
    memory_mb: int = 4096,
    files: dict[str, bytes] | None = None,
) -> ExecResult:
    """Run `code` as a script in a fresh temp dir; `files` are placed beside it."""
    with tempfile.TemporaryDirectory(prefix="kairyu-bench-") as tmp:
        workdir = Path(tmp)
        script = workdir / "main.py"
        script.write_text(code, encoding="utf-8")
        for name, data in (files or {}).items():
            (workdir / name).write_bytes(data)
        env = {"PATH": "/usr/bin:/bin", "HOME": tmp, "TMPDIR": tmp}
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(script)],
                input=stdin.encode(),
                capture_output=True,
                timeout=timeout_s,
                cwd=tmp,
                env=env,
                preexec_fn=_make_preexec(memory_mb, int(timeout_s) + 1),
                check=False,
            )
        except subprocess.TimeoutExpired as expired:
            return ExecResult(
                returncode=-1,
                stdout=(expired.stdout or b"").decode(errors="replace")[:_OUTPUT_CAP],
                stderr=(expired.stderr or b"").decode(errors="replace")[:_OUTPUT_CAP],
                timed_out=True,
            )
        return ExecResult(
            returncode=completed.returncode,
            stdout=completed.stdout.decode(errors="replace")[:_OUTPUT_CAP],
            stderr=completed.stderr.decode(errors="replace")[:_OUTPUT_CAP],
            timed_out=False,
        )


def has_module(name: str) -> bool:
    """Can the sandbox interpreter import `name`? (numpy for SciCode, etc.)"""
    result = run_python(f"import {name}\n", timeout_s=20.0)
    return result.ok
