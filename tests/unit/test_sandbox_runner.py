"""Sandbox runner report/validation logic (no Docker).

Silent mis-parsing here would convert failing tests into "passed" evidence fed
to the synthesis/verifier stages, so the summarization contract is pinned.
"""

import importlib.util
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "examples/qwen3.8-deepseek-v4-8gpu/sandbox/runner.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("sandbox_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _submission(**overrides):
    body = {
        "schema_version": 1,
        "language": "python",
        "files": {
            "solution.py": "x = 1",
            "test_solution.py": "def test_x():\n    assert True",
        },
        "entry": {"kind": "pytest"},
    }
    body.update(overrides)
    return body


def test_parse_rejects_non_python_language_as_422():
    with pytest.raises(runner.SubmissionError) as excinfo:
        runner.parse_submission(_submission(language="javascript"))
    assert excinfo.value.status_code == 422


def test_parse_rejects_path_escapes_and_requires_one_test_file():
    with pytest.raises(runner.SubmissionError):
        runner.parse_submission(_submission(files={"../evil.py": "x", "test_a.py": "y"}))
    with pytest.raises(runner.SubmissionError):
        runner.parse_submission(_submission(files={"solution.py": "x"}))
    with pytest.raises(runner.SubmissionError):
        runner.parse_submission(
            _submission(files={"test_a.py": "x", "test_b.py": "y", "solution.py": "z"})
        )
    parsed = runner.parse_submission(_submission())
    assert parsed["test_file"] == "test_solution.py"
    assert parsed["limits"]["wall_time_s"] == 10.0


def test_summarize_folds_pytest_report_counts():
    report = runner.summarize(
        {
            "exit_code": 1,
            "tests": [
                {"id": "test_solution.py::test_a", "outcome": "passed"},
                {"id": "test_solution.py::test_b", "outcome": "failed"},
                {"id": "test_solution.py::test_c", "outcome": "error"},
                {"id": 5, "outcome": "passed"},  # malformed row is dropped
            ],
        },
        timed_out=False,
        returncode=0,
        stdout=b"out",
        stderr=b"",
        output_bytes=1024,
        duration_ms=10,
    )
    assert report["status"] == "ok"
    assert (report["passed"], report["failed"], report["errors"]) == (1, 1, 1)
    assert len(report["tests"]) == 3


def test_summarize_missing_report_is_setup_error_never_ok():
    report = runner.summarize(
        None,
        timed_out=False,
        returncode=2,
        stdout=b"",
        stderr=b"import error",
        output_bytes=1024,
        duration_ms=5,
    )
    assert report["status"] == "setup_error"
    assert report["passed"] == 0


@pytest.mark.parametrize("exit_code", [2, 3, 4, 5])
def test_summarize_pytest_abnormal_exit_is_setup_error(exit_code):
    report = runner.summarize(
        {
            "exit_code": exit_code,
            "tests": [{"id": "test_solution.py::test_a", "outcome": "passed"}],
        },
        timed_out=False,
        returncode=0,
        stdout=b"",
        stderr=b"",
        output_bytes=1024,
        duration_ms=5,
    )

    assert report["status"] == "setup_error"
    assert report["detail"] == f"returncode=0; pytest_exit_code={exit_code}"


def test_summarize_shim_failure_with_report_is_setup_error():
    report = runner.summarize(
        {
            "exit_code": 0,
            "tests": [{"id": "test_solution.py::test_a", "outcome": "passed"}],
        },
        timed_out=False,
        returncode=-9,
        stdout=b"",
        stderr=b"",
        output_bytes=1024,
        duration_ms=5,
    )

    assert report["status"] == "setup_error"
    assert report["detail"] == "returncode=-9; pytest_exit_code=0"


def test_summarize_empty_test_report_is_setup_error():
    report = runner.summarize(
        {"exit_code": 0, "tests": []},
        timed_out=False,
        returncode=0,
        stdout=b"",
        stderr=b"",
        output_bytes=1024,
        duration_ms=5,
    )

    assert report["status"] == "setup_error"
    assert report["detail"] == "no_tests_collected"


def test_summarize_timeout_and_output_truncation():
    report = runner.summarize(
        {"exit_code": 0, "tests": []},
        timed_out=True,
        returncode=-9,
        stdout=b"a" * 100,
        stderr=b"",
        output_bytes=16,
        duration_ms=999,
    )
    assert report["status"] == "timeout"
    assert report["resource"]["timed_out"] is True
    assert report["resource"]["output_truncated"] is True
    assert report["stdout_excerpt"].endswith("[truncated]")


def test_unix_socket_server_healthcheck():
    # AF_UNIX paths are limited to 108 bytes on Linux; pytest's configured
    # temporary root can exceed that on WSL hosts.
    with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
        socket_path = Path(temp_dir) / "executor.sock"
        server = runner.ThreadingUnixHTTPServer(str(socket_path), runner.Handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            with socket.socket(socket.AF_UNIX) as client:
                client.settimeout(2)
                client.connect(str(socket_path))
                client.sendall(b"GET /healthz HTTP/1.0\r\nHost: executor\r\n\r\n")
                chunks = []
                while True:
                    chunk = client.recv(4096)
                    if not chunk:
                        break
                    chunks.append(chunk)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        response = b"".join(chunks)
        assert response.startswith(b"HTTP/1.0 200")
        assert b'{"status": "ok"}' in response


def test_communicate_capped_drains_both_pipes_without_retaining_full_output():
    output_bytes = 1024
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "sys.stdout.buffer.write(b'x' * 262144); "
                "sys.stdout.buffer.flush(); "
                "sys.stderr.buffer.write(b'y' * 262144)"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stdout, stderr = runner._communicate_capped(process, output_bytes)

    assert process.returncode == 0
    assert stdout == b"x" * (output_bytes + 1)
    assert stderr == b"y" * (output_bytes + 1)
    report = runner.summarize(
        {"exit_code": 0, "tests": []},
        timed_out=False,
        returncode=0,
        stdout=stdout,
        stderr=stderr,
        output_bytes=output_bytes,
        duration_ms=1,
    )
    assert report["resource"]["output_truncated"] is True


def test_sweep_kills_and_reaps_setsid_descendant_after_submission_exit():
    was_subreaper = runner._child_subreaper_enabled()
    child_pid = None
    runner._set_child_subreaper(True)
    try:
        parent = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import subprocess, sys; "
                    "child = subprocess.Popen("
                    "[sys.executable, '-c', 'import time; time.sleep(30)'], "
                    "start_new_session=True, stdout=subprocess.DEVNULL, "
                    "stderr=subprocess.DEVNULL); "
                    "print(child.pid, flush=True)"
                ),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = parent.communicate(timeout=5)
        assert parent.returncode == 0, stderr
        child_pid = int(stdout.strip())

        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            parents = runner._same_uid_process_parents(os.getuid())
            if parents.get(child_pid) == os.getpid():
                break
            time.sleep(0.01)
        assert parents.get(child_pid) == os.getpid()

        killed = runner._sweep_unowned_submission_processes(
            supervisor_pid=os.getpid(),
            target_uid=os.getuid(),
            active_roots=set(),
        )

        assert child_pid in killed
        assert not os.path.exists(f"/proc/{child_pid}")
        with pytest.raises(ChildProcessError):
            os.waitpid(child_pid, os.WNOHANG)
        child_pid = None
    finally:
        if child_pid is not None:
            try:
                os.kill(child_pid, 9)
                os.waitpid(child_pid, 0)
            except (ChildProcessError, ProcessLookupError):
                pass
        runner._set_child_subreaper(was_subreaper)


def test_run_submission_removes_setsid_child_before_return(monkeypatch):
    was_subreaper = runner._child_subreaper_enabled()
    child_pid = None
    runner._set_child_subreaper(True)
    try:
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temp_dir:
            root = Path(temp_dir)
            work_root = root / "work"
            work_root.mkdir()
            pid_path = root / "child.pid"
            marker_path = root / "escaped"
            child_code = (
                "import os, pathlib, time; "
                f"pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid())); "
                "time.sleep(0.5); "
                f"pathlib.Path({str(marker_path)!r}).write_text('escaped')"
            )
            test_code = (
                "import pathlib, subprocess, sys, time\n"
                "def test_spawn():\n"
                "    subprocess.Popen(\n"
                f"        [sys.executable, '-c', {child_code!r}],\n"
                "        start_new_session=True,\n"
                "        stdout=subprocess.DEVNULL,\n"
                "        stderr=subprocess.DEVNULL,\n"
                "    )\n"
                f"    path = pathlib.Path({str(pid_path)!r})\n"
                "    deadline = time.monotonic() + 2\n"
                "    while not path.exists() and time.monotonic() < deadline:\n"
                "        time.sleep(0.01)\n"
                "    assert path.exists()\n"
            )
            monkeypatch.setattr(runner, "WORK_ROOT", str(work_root))
            submission = runner.parse_submission(
                _submission(
                    files={"solution.py": "", "test_solution.py": test_code},
                    limits={"wall_time_s": 5, "processes": 1024},
                )
            )

            report = runner.run_submission(submission)
            child_pid = int(pid_path.read_text())

            assert report["status"] == "ok"
            assert not os.path.exists(f"/proc/{child_pid}")
            time.sleep(0.6)
            assert not marker_path.exists()
            child_pid = None
    finally:
        if child_pid is not None:
            try:
                os.kill(child_pid, 9)
                os.waitpid(child_pid, 0)
            except (ChildProcessError, ProcessLookupError):
                pass
        runner._set_child_subreaper(was_subreaper)
