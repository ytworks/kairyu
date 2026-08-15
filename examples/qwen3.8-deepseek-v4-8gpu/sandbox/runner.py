#!/usr/bin/env python3
"""Sandbox execution runner for the tiered coding example (ECO-D1/ECO-D3).

A single-file stdlib HTTP service that runs model-generated Python against
model-generated pytest files inside this already-isolated container
(networkless, read-only rootfs, non-root, no capabilities). The code under
execution is hostile by assumption: every submission runs in a fresh workdir
under a dedicated process group with rlimits, a cleared environment, an
in-process wall-clock killer, and capped output pipes — in addition to the
container-level limits enforced by compose.

Per-test outcomes are read back from a JSON report written by an in-sandbox
pytest plugin shim, never parsed from free text.
"""

from __future__ import annotations

import json
import os
import re
import resource
import shutil
import signal
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_CONCURRENCY = int(os.environ.get("RUNNER_MAX_CONCURRENCY", "4"))
WORK_ROOT = os.environ.get("RUNNER_WORK_ROOT", "/run/exec")
PORT = int(os.environ.get("RUNNER_PORT", "8080"))
SOCKET_PATH = os.environ.get("RUNNER_SOCKET_PATH")

_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_REPORT_NAME = "__kairyu_report__.json"
_SHIM_NAME = "__kairyu_shim__.py"

# Runs inside the sandboxed interpreter; per-test outcomes come from pytest's
# structured hook reports, written as JSON for the supervisor to read back.
_SHIM = """\
import json
import sys

import pytest


class _Collector:
    def __init__(self):
        self.tests = []
        self._failed_setup = set()

    def pytest_runtest_logreport(self, report):
        if report.when == "setup" and report.outcome != "passed":
            self._failed_setup.add(report.nodeid)
            self.tests.append({"id": report.nodeid, "outcome": "error"})
        elif report.when == "call" and report.nodeid not in self._failed_setup:
            outcome = "passed" if report.outcome == "passed" else "failed"
            self.tests.append({"id": report.nodeid, "outcome": outcome})


collector = _Collector()
exit_code = pytest.main(
    ["-q", "-p", "no:cacheprovider", "--maxfail=100", TEST_FILE],
    plugins=[collector],
)
with open(REPORT_FILE, "w", encoding="utf-8") as handle:
    json.dump({"exit_code": int(exit_code), "tests": collector.tests}, handle)
"""


class SubmissionError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def parse_submission(body: object) -> dict:
    """Validate an /v1/execute body; raise SubmissionError on rejection."""

    if not isinstance(body, dict):
        raise SubmissionError(400, "body must be a JSON object")
    if body.get("language") != "python":
        raise SubmissionError(422, "unsupported_language")
    entry = body.get("entry")
    if not isinstance(entry, dict) or entry.get("kind") != "pytest":
        raise SubmissionError(400, "entry.kind must be 'pytest'")
    files = body.get("files")
    if not isinstance(files, dict) or not files:
        raise SubmissionError(400, "files must be a non-empty object")
    if len(files) > 16:
        raise SubmissionError(400, "too many files")
    test_files = []
    for name, content in files.items():
        if not isinstance(name, str) or not _FILENAME.match(name) or ".." in name:
            raise SubmissionError(400, f"invalid file name {name!r}")
        if not isinstance(content, str) or len(content) > 262_144:
            raise SubmissionError(400, f"invalid file content for {name!r}")
        if name.startswith("test_"):
            test_files.append(name)
    if len(test_files) != 1:
        raise SubmissionError(400, "exactly one test_*.py file is required")
    raw_limits = body.get("limits") or {}
    if not isinstance(raw_limits, dict):
        raise SubmissionError(400, "limits must be an object")

    def limit(key: str, default: float, low: float, high: float) -> float:
        value = raw_limits.get(key, default)
        if not isinstance(value, (int, float)) or not low <= value <= high:
            raise SubmissionError(400, f"limit {key} out of range")
        return float(value)

    return {
        "files": dict(files),
        "test_file": test_files[0],
        "limits": {
            "wall_time_s": limit("wall_time_s", 10.0, 0.5, 120.0),
            "cpu_time_s": limit("cpu_time_s", 8.0, 0.5, 120.0),
            "memory_mb": int(limit("memory_mb", 512, 32, 4096)),
            "processes": int(limit("processes", 64, 1, 1024)),
            "output_bytes": int(limit("output_bytes", 32768, 1024, 1_048_576)),
        },
    }


def summarize(
    report_payload: object,
    *,
    timed_out: bool,
    returncode: int | None,
    stdout: bytes,
    stderr: bytes,
    output_bytes: int,
    duration_ms: int,
) -> dict:
    """Fold the shim's JSON report and process outcome into the wire report."""

    def excerpt(raw: bytes) -> tuple[str, bool]:
        truncated = len(raw) > output_bytes
        text = raw[:output_bytes].decode("utf-8", errors="replace")
        return (text + "\n[truncated]" if truncated else text), truncated

    stdout_text, stdout_truncated = excerpt(stdout)
    stderr_text, stderr_truncated = excerpt(stderr)
    base = {
        "schema_version": 1,
        "stdout_excerpt": stdout_text,
        "stderr_excerpt": stderr_text,
        "duration_ms": duration_ms,
        "resource": {
            "timed_out": timed_out,
            "output_truncated": stdout_truncated or stderr_truncated,
        },
    }
    if timed_out:
        return {**base, "status": "timeout", "tests": [], "passed": 0, "failed": 0, "errors": 0}
    tests = []
    if isinstance(report_payload, dict):
        for entry in report_payload.get("tests", ()):
            if (
                isinstance(entry, dict)
                and isinstance(entry.get("id"), str)
                and entry.get("outcome") in {"passed", "failed", "error"}
            ):
                tests.append({"id": entry["id"], "outcome": entry["outcome"]})
    if report_payload is None or not isinstance(report_payload, dict):
        return {
            **base,
            "status": "setup_error",
            "tests": [],
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "detail": f"returncode={returncode}",
        }
    passed = sum(1 for entry in tests if entry["outcome"] == "passed")
    failed = sum(1 for entry in tests if entry["outcome"] == "failed")
    errors = sum(1 for entry in tests if entry["outcome"] == "error")
    return {
        **base,
        "status": "ok",
        "tests": tests,
        "passed": passed,
        "failed": failed,
        "errors": errors,
    }


def _preexec(limits: dict):
    def apply() -> None:
        os.setsid()
        cpu = int(limits["cpu_time_s"]) + 1
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        memory = limits["memory_mb"] * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        processes = limits["processes"]
        resource.setrlimit(resource.RLIMIT_NPROC, (processes, processes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024 * 1024, 8 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    return apply


def _communicate_capped(process: subprocess.Popen[bytes], output_bytes: int) -> tuple[bytes, bytes]:
    """Drain both pipes concurrently while retaining only cap + 1 bytes."""

    assert process.stdout is not None
    assert process.stderr is not None
    retained = [bytearray(), bytearray()]
    retain_bytes = output_bytes + 1

    def drain(index: int, stream) -> None:
        while chunk := stream.read(64 * 1024):
            remaining = retain_bytes - len(retained[index])
            if remaining > 0:
                retained[index].extend(chunk[:remaining])

    readers = [
        threading.Thread(target=drain, args=(0, process.stdout)),
        threading.Thread(target=drain, args=(1, process.stderr)),
    ]
    for reader in readers:
        reader.start()
    process.wait()
    for reader in readers:
        reader.join()
    return bytes(retained[0]), bytes(retained[1])


def run_submission(submission: dict) -> dict:
    """Execute one validated submission and return the wire report."""

    limits = submission["limits"]
    started = time.monotonic()
    workdir = tempfile.mkdtemp(prefix="exec-", dir=WORK_ROOT)
    try:
        for name, content in submission["files"].items():
            with open(os.path.join(workdir, name), "w", encoding="utf-8") as handle:
                handle.write(content)
        report_path = os.path.join(workdir, _REPORT_NAME)
        shim = f"TEST_FILE = {submission['test_file']!r}\nREPORT_FILE = {report_path!r}\n{_SHIM}"
        shim_path = os.path.join(workdir, _SHIM_NAME)
        with open(shim_path, "w", encoding="utf-8") as handle:
            handle.write(shim)
        env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": workdir,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
        }
        process = subprocess.Popen(
            [sys.executable, shim_path],
            cwd=workdir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=_preexec(limits),
            start_new_session=False,  # _preexec owns setsid
        )
        timed_out = False

        def kill_group() -> None:
            nonlocal timed_out
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        timer = threading.Timer(limits["wall_time_s"], kill_group)
        timer.start()
        try:
            stdout, stderr = _communicate_capped(process, limits["output_bytes"])
        finally:
            timer.cancel()
        report_payload = None
        if not timed_out and os.path.exists(report_path):
            try:
                with open(report_path, encoding="utf-8") as handle:
                    report_payload = json.load(handle)
            except (OSError, ValueError):
                report_payload = None
        return summarize(
            report_payload,
            timed_out=timed_out,
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
            output_bytes=limits["output_bytes"],
            duration_ms=int((time.monotonic() - started) * 1000),
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


_slots = threading.BoundedSemaphore(MAX_CONCURRENCY)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # no request logging: hostile inputs
        return

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - http.server contract
        if self.path == "/healthz":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802 - http.server contract
        if self.path != "/v1/execute":
            self._send_json(404, {"error": "not_found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > 4 * 1024 * 1024:
            self._send_json(400, {"error": "invalid_content_length"})
            return
        try:
            body = json.loads(self.rfile.read(length))
        except ValueError:
            self._send_json(400, {"error": "invalid_json"})
            return
        try:
            submission = parse_submission(body)
        except SubmissionError as error:
            self._send_json(error.status_code, {"error": str(error)})
            return
        if not _slots.acquire(blocking=False):
            self._send_json(429, {"error": "busy"})
            return
        try:
            report = run_submission(submission)
        except Exception:
            self._send_json(500, {"error": "runner_error"})
            return
        finally:
            _slots.release()
        self._send_json(200, report)


class ThreadingUnixHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def main() -> None:
    os.makedirs(WORK_ROOT, exist_ok=True)
    socket_path = SOCKET_PATH
    if socket_path:
        os.makedirs(os.path.dirname(socket_path), exist_ok=True)
        try:
            os.unlink(socket_path)
        except FileNotFoundError:
            pass
        server = ThreadingUnixHTTPServer(socket_path, Handler)
        os.chmod(socket_path, 0o660)
    else:
        server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if socket_path:
            try:
                os.unlink(socket_path)
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    main()
