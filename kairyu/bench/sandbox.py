"""Subprocess sandbox for execution-scored benchmarks (LiveCodeBench, SciCode).

Isolation = fresh temp cwd, scrubbed env, `python -I`, rlimits (address space /
CPU / file size), a NEW SESSION so the whole process tree is one group, and a
wall-clock kill that reaps the ENTIRE group (grandchildren included). This is a
guard against runaway benchmark code, NOT a security boundary against a hostile
model — documented in docs/benchmarks.md; a container runner is a future hook.

Memory rlimits are best-effort: macOS rejects RLIMIT_AS / RLIMIT_DATA with
EINVAL, so there the wall-clock kill is the only containment. Linux (CI and
deploy) enforces them. Fork-bomb containment is the group kill, not a fixed
RLIMIT_NPROC (a per-user cap that made scores host-dependent — a busy host would
EAGAIN legit multiprocessing while a quiet one passed).
"""

from __future__ import annotations

import os
import signal
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
        os.setsid()  # own session/group so the wall-clock kill can reap the tree
        limit = memory_mb * 1024 * 1024
        for memory_rlimit in (resource.RLIMIT_AS, resource.RLIMIT_DATA):
            try:
                resource.setrlimit(memory_rlimit, (limit, limit))
                break
            except (ValueError, OSError):
                continue  # macOS rejects address-space rlimits with EINVAL
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_FSIZE, (32_000_000, 32_000_000))

    return preexec


def _reap_group(pid: int) -> None:
    """Best-effort SIGKILL of the child's whole process group (grandchildren)."""
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


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
        process = subprocess.Popen(
            [sys.executable, "-I", str(script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=tmp,
            env=env,
            preexec_fn=_make_preexec(memory_mb, int(timeout_s) + 1),
        )
        try:
            stdout, stderr = process.communicate(input=stdin.encode(), timeout=timeout_s)
            timed_out = False
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            _reap_group(process.pid)  # kill grandchildren too, not just the child
            stdout, stderr = process.communicate()
            timed_out = True
            returncode = -1
        finally:
            _reap_group(process.pid)  # reap any lingering forked descendants
        return ExecResult(
            returncode=returncode,
            stdout=(stdout or b"").decode(errors="replace")[:_OUTPUT_CAP],
            stderr=(stderr or b"").decode(errors="replace")[:_OUTPUT_CAP],
            timed_out=timed_out,
        )


def has_module(name: str) -> bool:
    """Can the sandbox interpreter import `name`? (numpy for SciCode, etc.)"""
    result = run_python(f"import {name}\n", timeout_s=20.0)
    return result.ok
