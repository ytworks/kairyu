from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import profile_server


class _CompletedProcess:
    pid = 123

    def __init__(self, returncode: int, *, on_wait=None) -> None:
        self.returncode: int | None = None
        self._final_returncode = returncode
        self._on_wait = on_wait

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self._on_wait is not None:
            callback, self._on_wait = self._on_wait, None
            callback()
        self.returncode = self._final_returncode
        return self.returncode


def _nsys_invocation(tmp_path: Path) -> profile_server.ProfileInvocation:
    return profile_server.build_invocation(
        "nsys",
        executable="/tools/nsys",
        output=tmp_path / "server.nsys-rep",
        duration=5,
        serve_args=("--", "deploy/gpu.yaml"),
    )


def _write_nsys_capture(
    invocation: profile_server.ProfileInvocation,
    *,
    span_ns: int,
    use_start_stop: bool = False,
) -> None:
    invocation.artifact.write_bytes(b"nsys report")
    sqlite_path = next(
        path for path in invocation.reserved_outputs if path.suffix == ".sqlite"
    )
    with sqlite3.connect(sqlite_path) as connection:
        if use_start_stop:
            connection.execute(
                "CREATE TABLE ANALYSIS_DETAILS(startTime INTEGER, stopTime INTEGER)"
            )
            connection.execute(
                "INSERT INTO ANALYSIS_DETAILS VALUES (?, ?)", (100, 100 + span_ns)
            )
        else:
            connection.execute("CREATE TABLE ANALYSIS_DETAILS(duration INTEGER)")
            connection.execute("INSERT INTO ANALYSIS_DETAILS VALUES (?)", (span_ns,))


def _install_completed_process(
    monkeypatch,
    process: _CompletedProcess,
    *,
    cleanup: profile_server.ProcessGroupCleanup | None = None,
) -> None:
    monkeypatch.setattr(
        profile_server.subprocess,
        "Popen",
        lambda _argv, *, start_new_session: process,
    )
    monkeypatch.setattr(
        profile_server,
        "_cleanup_process_group",
        lambda _process_group: cleanup
        or profile_server.ProcessGroupCleanup(
            existed=False,
            complete=True,
            kill_required=False,
        ),
    )


def _record_live_pyspy_target(
    monkeypatch,
    invocation: profile_server.ProfileInvocation,
    *,
    pid: int = 456,
    process_group: int = _CompletedProcess.pid,
    write_artifact: bool = True,
    samples: bool = True,
) -> profile_server.ProcessIdentity:
    assert invocation.target_started is not None
    identity = profile_server.ProcessIdentity(
        pid=pid,
        process_group=process_group,
        state="S",
        start_time_ticks=987_654,
    )
    invocation.target_started.parent.mkdir(parents=True, exist_ok=True)
    invocation.target_started.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pid": identity.pid,
                "process_group": identity.process_group,
                "start_time_ticks": identity.start_time_ticks,
            }
        ),
        encoding="utf-8",
    )
    if write_artifact:
        invocation.artifact.write_text(
            json.dumps(
                {
                    "$schema": "https://www.speedscope.app/file-format-schema.json",
                    "profiles": [
                        {
                            "type": "sampled",
                            "name": f'Process {pid} Thread {pid} "MainThread"',
                            "samples": [[0]] if samples else [],
                            "weights": [0.01] if samples else [],
                        }
                    ],
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
    monkeypatch.setattr(
        profile_server,
        "_read_process_identity",
        lambda candidate: identity if candidate == pid else None,
    )
    return identity


def test_nsys_invocation_does_not_elevate_and_forwards_serve_args_verbatim() -> None:
    invocation = profile_server.build_invocation(
        "nsys",
        executable="/tools/nsys",
        output=Path("artifacts/server trace.nsys-rep"),
        duration=45,
        delay=90,
        serve_args=(
            "--",
            "configs/model with spaces.yaml",
            "--host",
            "127.0.0.1",
            "--generation-config",
            "none;still-one-argument",
        ),
        python="/venv/bin/python",
    )

    assert invocation.artifact == Path("artifacts/server trace.nsys-rep")
    assert invocation.reserved_outputs == (
        Path("artifacts/server trace.nsys-rep"),
        Path("artifacts/server trace.sqlite"),
    )
    assert invocation.argv == (
        "/tools/nsys",
        "profile",
        "--trace=cuda,nvtx,osrt,python-gil",
        "--sample=process-tree",
        "--cpuctxsw=process-tree",
        "--python-sampling=true",
        "--python-sampling-frequency=99",
        "--cuda-trace-scope=process-tree",
        "--wait=all",
        "--kill=sigterm",
        "--stats=true",
        "--discard-environment=true",
        "--force-overwrite=false",
        "--output=artifacts/server trace",
        "--delay=90",
        "--duration=45",
        "/venv/bin/python",
        "-m",
        "kairyu.entrypoints.cli",
        "serve",
        "configs/model with spaces.yaml",
        "--host",
        "127.0.0.1",
        "--generation-config",
        "none;still-one-argument",
    )
    assert "sudo" not in invocation.argv


def test_py_spy_launches_python_parent_and_profiles_its_subprocesses() -> None:
    invocation = profile_server.build_invocation(
        "py-spy",
        executable="/tools/py-spy",
        output=Path("artifacts/api.speedscope.json"),
        duration=60,
        serve_args=("--", "deploy/gpu.yaml", "--port", "8100"),
        python="/venv/bin/python",
    )

    assert invocation.artifact == Path("artifacts/api.speedscope.json")
    assert invocation.target_started == Path(
        "artifacts/api.speedscope.json.target-started.json"
    )
    assert invocation.reserved_outputs == (
        Path("artifacts/api.speedscope.json"),
        Path("artifacts/api.speedscope.json.target-started.json"),
    )
    assert invocation.target_status == Path(
        "artifacts/api.speedscope.json.target-exit.json"
    )
    assert invocation.collision_outputs == (
        Path("artifacts/api.speedscope.json"),
        Path("artifacts/api.speedscope.json.target-started.json"),
        Path("artifacts/api.speedscope.json.target-exit.json"),
    )
    assert invocation.argv == (
        "/tools/py-spy",
        "record",
        "--subprocesses",
        "--threads",
        "--format=speedscope",
        "--output=artifacts/api.speedscope.json",
        "--duration=60",
        "--",
        "/venv/bin/python",
        str(Path(profile_server.__file__).resolve()),
        "_supervise",
        "--started",
        "artifacts/api.speedscope.json.target-started.json",
        "--status",
        "artifacts/api.speedscope.json.target-exit.json",
        "--",
        "/venv/bin/python",
        "-m",
        "kairyu.entrypoints.cli",
        "serve",
        "deploy/gpu.yaml",
        "--port",
        "8100",
    )


@pytest.mark.parametrize("profiler", ["nsys", "py-spy"])
def test_missing_profiler_fails_before_creating_output(
    profiler, monkeypatch, tmp_path, capsys
) -> None:
    output = tmp_path / "new" / (
        "trace.nsys-rep" if profiler == "nsys" else "trace.speedscope.json"
    )
    monkeypatch.setattr(profile_server.shutil, "which", lambda _name: None)

    result = profile_server.main(
        [
            profiler,
            "--output",
            str(output),
            "--duration",
            "30",
            "--",
            "deploy/gpu.yaml",
        ]
    )

    assert result == profile_server.NOT_RUN_EXIT
    assert not output.parent.exists()
    assert f"required executable not found: {profiler}" in capsys.readouterr().err


def test_main_creates_artifact_parent_then_runs_exact_argv(
    monkeypatch, tmp_path
) -> None:
    output = tmp_path / "profiles" / "server.speedscope.json"
    expected_invocation = profile_server.build_invocation(
        "py-spy",
        executable="/tools/py-spy",
        output=output,
        duration=10,
        serve_args=(
            "--",
            "deploy/gpu.yaml",
            "--host",
            "host with spaces",
        ),
    )
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        profile_server.shutil,
        "which",
        lambda name: f"/tools/{name}",
    )

    def fake_popen(argv, *, start_new_session):
        seen["argv"] = tuple(argv)
        seen["start_new_session"] = start_new_session
        return _CompletedProcess(
            0,
            on_wait=lambda: _record_live_pyspy_target(
                monkeypatch, expected_invocation
            ),
        )

    monkeypatch.setattr(profile_server.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        profile_server,
        "_cleanup_process_group",
        lambda _process_group: profile_server.ProcessGroupCleanup(
            existed=True,
            complete=True,
            kill_required=False,
        ),
    )
    times = iter((100.0, 110.0))
    monkeypatch.setattr(profile_server.time, "monotonic", lambda: next(times))

    result = profile_server.main(
        [
            "py-spy",
            "--output",
            str(output),
            "--duration",
            "10",
            "--",
            "deploy/gpu.yaml",
            "--host",
            "host with spaces",
        ]
    )

    assert result == 0
    assert output.parent.is_dir()
    assert seen["start_new_session"] is True
    assert seen["argv"] == expected_invocation.argv


def test_existing_artifact_is_not_overwritten(monkeypatch, tmp_path, capsys) -> None:
    output = tmp_path / "server.nsys-rep"
    output.write_text("retained evidence", encoding="utf-8")
    monkeypatch.setattr(profile_server.shutil, "which", lambda _name: "/tools/nsys")
    monkeypatch.setattr(
        profile_server.subprocess,
        "Popen",
        lambda *_args: pytest.fail("existing evidence must not be overwritten"),
    )

    result = profile_server.main(
        [
            "nsys",
            "--output",
            str(output),
            "--duration",
            "30",
            "--",
            "deploy/gpu.yaml",
        ]
    )

    assert result == profile_server.NOT_RUN_EXIT
    assert output.read_text(encoding="utf-8") == "retained evidence"
    assert "output artifact already exists" in capsys.readouterr().err


def test_missing_deployment_spec_argument_fails_without_exec(monkeypatch) -> None:
    monkeypatch.setattr(profile_server.shutil, "which", lambda _name: "/tools/nsys")
    monkeypatch.setattr(
        profile_server.subprocess,
        "Popen",
        lambda *_args: pytest.fail("invalid command must not execute"),
    )

    assert (
        profile_server.main(
            [
                "nsys",
                "--output",
                "/tmp/profile.nsys-rep",
                "--duration",
                "30",
                "--",
            ]
        )
        == profile_server.NOT_RUN_EXIT
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["nsys", "--output", "/tmp/profile.nsys-rep", "--", "deploy/gpu.yaml"],
        [
            "nsys",
            "--output",
            "/tmp/profile.nsys-rep",
            "--duration",
            "0",
            "--",
            "deploy/gpu.yaml",
        ],
        [
            "nsys",
            "--output",
            "/tmp/profile.nsys-rep",
            "--duration",
            "-1",
            "--",
            "deploy/gpu.yaml",
        ],
        [
            "nsys",
            "--output",
            "/tmp/profile.nsys-rep",
            "--duration",
            "3601",
            "--",
            "deploy/gpu.yaml",
        ],
    ],
    ids=("missing", "zero", "negative", "over-limit"),
)
def test_duration_is_required_positive_and_bounded(argv) -> None:
    with pytest.raises(SystemExit) as exc_info:
        profile_server.main(argv)

    assert exc_info.value.code == 2


@pytest.mark.parametrize("delay", ["-1", "3601"])
def test_nsys_delay_is_non_negative_and_bounded(delay) -> None:
    with pytest.raises(SystemExit) as exc_info:
        profile_server.main(
            [
                "nsys",
                "--output",
                "/tmp/profile.nsys-rep",
                "--duration",
                "30",
                "--delay",
                delay,
                "--",
                "deploy/gpu.yaml",
            ]
        )

    assert exc_info.value.code == 2


def test_nsys_companion_sqlite_collision_is_rejected(
    monkeypatch, tmp_path, capsys
) -> None:
    output = tmp_path / "server.nsys-rep"
    sqlite = tmp_path / "server.sqlite"
    sqlite.write_text("retained export", encoding="utf-8")
    monkeypatch.setattr(profile_server.shutil, "which", lambda _name: "/tools/nsys")
    monkeypatch.setattr(
        profile_server.subprocess,
        "Popen",
        lambda *_args: pytest.fail("companion evidence must not be overwritten"),
    )

    result = profile_server.main(
        [
            "nsys",
            "--output",
            str(output),
            "--duration",
            "30",
            "--",
            "deploy/gpu.yaml",
        ]
    )

    assert result == profile_server.NOT_RUN_EXIT
    assert "server.sqlite" in capsys.readouterr().err


def test_py_spy_output_is_reserved_atomically_with_private_permissions(
    tmp_path,
) -> None:
    output = tmp_path / "server.speedscope.json"

    identity = profile_server._reserve_output(output)

    metadata = output.stat()
    assert identity == (metadata.st_dev, metadata.st_ino)
    assert metadata.st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        profile_server._reserve_output(output)


def test_py_spy_reservation_race_fails_without_launch(
    monkeypatch, tmp_path, capsys
) -> None:
    output = tmp_path / "server.speedscope.json"
    monkeypatch.setattr(profile_server.shutil, "which", lambda _name: "/tools/py-spy")
    monkeypatch.setattr(
        profile_server,
        "_reserve_output",
        lambda _path: (_ for _ in ()).throw(FileExistsError()),
    )
    monkeypatch.setattr(
        profile_server.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("racing output must not launch"),
    )

    result = profile_server.main(
        [
            "py-spy",
            "--output",
            str(output),
            "--duration",
            "30",
            "--",
            "deploy/gpu.yaml",
        ]
    )

    assert result == profile_server.NOT_RUN_EXIT
    assert "already exists" in capsys.readouterr().err


def test_failed_py_spy_run_removes_only_its_empty_reservation(
    monkeypatch, tmp_path
) -> None:
    output = tmp_path / "server.speedscope.json"
    monkeypatch.setattr(profile_server.shutil, "which", lambda _name: "/tools/py-spy")
    monkeypatch.setattr(
        profile_server,
        "_run_invocation",
        lambda *_args, **_kwargs: profile_server.INVALID_CAPTURE_EXIT,
    )

    result = profile_server.main(
        [
            "py-spy",
            "--output",
            str(output),
            "--duration",
            "30",
            "--",
            "deploy/gpu.yaml",
        ]
    )

    assert result == profile_server.INVALID_CAPTURE_EXIT
    assert not output.exists()


def test_target_status_collision_is_rejected_before_launch(
    monkeypatch, tmp_path, capsys
) -> None:
    output = tmp_path / "server.speedscope.json"
    target_status = Path(f"{output}.target-exit.json")
    target_status.write_text("retained", encoding="utf-8")
    monkeypatch.setattr(profile_server.shutil, "which", lambda _name: "/tools/py-spy")
    monkeypatch.setattr(
        profile_server.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("status collision must not launch"),
    )

    result = profile_server.main(
        [
            "py-spy",
            "--output",
            str(output),
            "--duration",
            "30",
            "--",
            "deploy/gpu.yaml",
        ]
    )

    assert result == profile_server.NOT_RUN_EXIT
    assert target_status.read_text(encoding="utf-8") == "retained"
    assert "target-exit.json" in capsys.readouterr().err


def test_output_directory_error_is_reported_without_traceback(
    monkeypatch, tmp_path, capsys
) -> None:
    output = tmp_path / "blocked" / "server.speedscope.json"
    monkeypatch.setattr(profile_server.shutil, "which", lambda _name: "/tools/py-spy")
    monkeypatch.setattr(
        Path,
        "mkdir",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError()),
    )

    result = profile_server.main(
        [
            "py-spy",
            "--output",
            str(output),
            "--duration",
            "30",
            "--",
            "deploy/gpu.yaml",
        ]
    )

    assert result == profile_server.NOT_RUN_EXIT
    assert "could not prepare output directory: PermissionError" in capsys.readouterr().err


def test_dry_run_needs_no_profiler_and_prints_exact_quoted_argv(
    monkeypatch, tmp_path, capsys
) -> None:
    output = tmp_path / "not-created" / "server.speedscope.json"
    monkeypatch.setattr(profile_server.shutil, "which", lambda _name: None)

    result = profile_server.main(
        [
            "py-spy",
            "--output",
            str(output),
            "--duration",
            "30",
            "--dry-run",
            "--",
            "config with spaces.yaml",
        ]
    )

    printed = capsys.readouterr().out
    assert result == 0
    assert not output.parent.exists()
    assert "diagnostic-only profiling artifact" in printed
    assert printed.splitlines()[1].startswith("+ py-spy record --subprocesses")
    assert "'config with spaces.yaml'" in printed


def test_nsys_dry_run_includes_delayed_collection_window(monkeypatch, capsys) -> None:
    monkeypatch.setattr(profile_server.shutil, "which", lambda _name: None)

    result = profile_server.main(
        [
            "nsys",
            "--output",
            "/tmp/server.nsys-rep",
            "--duration",
            "60",
            "--delay",
            "240",
            "--dry-run",
            "--",
            "deploy/gpu.yaml",
        ]
    )

    assert result == 0
    assert "--delay=240 --duration=60" in capsys.readouterr().out


def test_supervisor_forwards_exact_target_and_records_an_early_exit(
    monkeypatch, tmp_path
) -> None:
    started = tmp_path / "target-started.json"
    status = tmp_path / "target-exit.json"
    seen: dict[str, tuple[str, ...]] = {}
    identity = profile_server.ProcessIdentity(
        pid=_CompletedProcess.pid,
        process_group=os.getpgrp(),
        state="R",
        start_time_ticks=321,
    )

    def fake_popen(argv):
        seen["argv"] = tuple(argv)
        return _CompletedProcess(17)

    monkeypatch.setattr(profile_server.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        profile_server,
        "_read_process_identity",
        lambda _pid: identity,
    )
    times = iter((1_000, 2_500))
    monkeypatch.setattr(profile_server.time, "monotonic_ns", lambda: next(times))

    result = profile_server._supervise_target(
        [
            "--started",
            str(started),
            "--status",
            str(status),
            "--",
            "/venv/bin/python",
            "argument with spaces",
            "semi;colon",
        ]
    )

    assert result == 17
    assert seen["argv"] == (
        "/venv/bin/python",
        "argument with spaces",
        "semi;colon",
    )
    assert json.loads(status.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "returncode": 17,
        "elapsed_ns": 1_500,
    }
    assert json.loads(started.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "pid": identity.pid,
        "process_group": identity.process_group,
        "start_time_ticks": identity.start_time_ticks,
    }


def test_supervisor_status_write_never_overwrites_a_racing_file(
    monkeypatch, tmp_path
) -> None:
    started = tmp_path / "target-started.json"
    status = tmp_path / "target-exit.json"
    status.write_text("retained", encoding="utf-8")
    monkeypatch.setattr(
        profile_server.subprocess,
        "Popen",
        lambda _argv: _CompletedProcess(0),
    )
    monkeypatch.setattr(
        profile_server,
        "_read_process_identity",
        lambda pid: profile_server.ProcessIdentity(
            pid=pid,
            process_group=os.getpgrp(),
            state="R",
            start_time_ticks=654,
        ),
    )
    times = iter((1_000, 2_000))
    monkeypatch.setattr(profile_server.time, "monotonic_ns", lambda: next(times))

    result = profile_server._supervise_target(
        [
            "--started",
            str(started),
            "--status",
            str(status),
            "--",
            "/venv/bin/python",
        ]
    )

    assert result == 0
    assert status.read_text(encoding="utf-8") == "retained"


def test_process_group_cleanup_terms_residual_grandchildren(monkeypatch) -> None:
    alive = True
    calls: list[int] = []

    def fake_killpg(_process_group, signum):
        nonlocal alive
        calls.append(signum)
        if signum == 0:
            if not alive:
                raise ProcessLookupError
        elif signum == signal.SIGTERM:
            alive = False

    monkeypatch.setattr(profile_server.os, "killpg", fake_killpg)

    result = profile_server._cleanup_process_group(999)

    assert result == profile_server.ProcessGroupCleanup(
        existed=True,
        complete=True,
        kill_required=False,
    )
    assert calls == [0, signal.SIGTERM, 0]


def test_process_group_cleanup_reports_when_sigkill_was_required(
    monkeypatch,
) -> None:
    sent: list[int] = []
    waits = iter((False, True))
    monkeypatch.setattr(profile_server, "_process_group_exists", lambda _group: True)
    monkeypatch.setattr(
        profile_server,
        "_send_process_group",
        lambda _group, signum: sent.append(signum) or True,
    )
    monkeypatch.setattr(
        profile_server,
        "_wait_process_group_exit",
        lambda _group, _timeout: next(waits),
    )

    result = profile_server._cleanup_process_group(999)

    assert result == profile_server.ProcessGroupCleanup(
        existed=True,
        complete=True,
        kill_required=True,
    )
    assert sent == [signal.SIGTERM, signal.SIGKILL]


@pytest.mark.parametrize("returncode", [0, 128 + signal.SIGTERM])
def test_nsys_accepts_success_only_with_a_complete_sqlite_trace_span(
    returncode, monkeypatch, tmp_path
) -> None:
    invocation = _nsys_invocation(tmp_path)
    _write_nsys_capture(invocation, span_ns=5_000_000_000)
    _install_completed_process(monkeypatch, _CompletedProcess(returncode))
    times = iter((10.0, 15.0))
    monkeypatch.setattr(profile_server.time, "monotonic", lambda: next(times))

    result = profile_server._run_invocation(
        invocation,
        profiler="nsys",
        duration=5,
        delay=0,
    )

    assert result == 0


def test_nsys_uses_start_stop_span_when_duration_column_is_absent(
    monkeypatch, tmp_path
) -> None:
    invocation = _nsys_invocation(tmp_path)
    _write_nsys_capture(
        invocation,
        span_ns=5_000_000_000,
        use_start_stop=True,
    )
    _install_completed_process(
        monkeypatch,
        _CompletedProcess(0),
        cleanup=profile_server.ProcessGroupCleanup(
            existed=True,
            complete=True,
            kill_required=False,
        ),
    )
    times = iter((10.0, 15.0))
    monkeypatch.setattr(profile_server.time, "monotonic", lambda: next(times))

    assert (
        profile_server._run_invocation(
            invocation,
            profiler="nsys",
            duration=5,
            delay=0,
        )
        == 0
    )


def test_late_report_generation_cannot_hide_an_early_nsys_sigterm(
    monkeypatch, tmp_path
) -> None:
    invocation = _nsys_invocation(tmp_path)
    _write_nsys_capture(invocation, span_ns=4_999_999_999)
    _install_completed_process(
        monkeypatch,
        _CompletedProcess(128 + signal.SIGTERM),
    )
    times = iter((10.0, 10_000.0))
    monkeypatch.setattr(profile_server.time, "monotonic", lambda: next(times))

    result = profile_server._run_invocation(
        invocation,
        profiler="nsys",
        duration=5,
        delay=0,
    )

    assert result == 128 + signal.SIGTERM


@pytest.mark.parametrize(
    ("returncode", "expected"),
    [
        (0, profile_server.INVALID_CAPTURE_EXIT),
        (128 + signal.SIGTERM, 128 + signal.SIGTERM),
    ],
)
@pytest.mark.parametrize(
    "broken_output", ["report", "sqlite", "empty-sqlite", "corrupt-sqlite"]
)
def test_nsys_rejects_missing_empty_or_invalid_outputs(
    broken_output, returncode, expected, monkeypatch, tmp_path
) -> None:
    invocation = _nsys_invocation(tmp_path)
    _write_nsys_capture(invocation, span_ns=5_000_000_000)
    sqlite_path = next(
        path for path in invocation.reserved_outputs if path.suffix == ".sqlite"
    )
    if broken_output == "report":
        invocation.artifact.unlink()
    elif broken_output == "sqlite":
        sqlite_path.unlink()
    elif broken_output == "empty-sqlite":
        sqlite_path.write_bytes(b"")
    else:
        sqlite_path.write_bytes(b"not a SQLite database")
    _install_completed_process(monkeypatch, _CompletedProcess(returncode))
    times = iter((10.0, 15.0))
    monkeypatch.setattr(profile_server.time, "monotonic", lambda: next(times))

    result = profile_server._run_invocation(
        invocation,
        profiler="nsys",
        duration=5,
        delay=0,
    )

    assert result == expected


@pytest.mark.parametrize(
    ("elapsed", "write_output", "target_exited", "expected"),
    [
        (5.0, True, False, 0),
        (4.5, True, False, 0),
        (4.499, True, False, profile_server.INVALID_CAPTURE_EXIT),
        (5.0, False, False, profile_server.INVALID_CAPTURE_EXIT),
        (5.0, True, True, profile_server.INVALID_CAPTURE_EXIT),
    ],
    ids=(
        "complete",
        "timer-tolerance",
        "early",
        "missing-output",
        "target-exited",
    ),
)
def test_py_spy_zero_exit_requires_duration_and_nonempty_output(
    elapsed, write_output, target_exited, expected, monkeypatch, tmp_path
) -> None:
    output = tmp_path / "server.speedscope.json"
    invocation = profile_server.build_invocation(
        "py-spy",
        executable="/tools/py-spy",
        output=output,
        duration=5,
        serve_args=("--", "deploy/gpu.yaml"),
    )
    _record_live_pyspy_target(
        monkeypatch,
        invocation,
        write_artifact=write_output,
    )
    if target_exited:
        assert invocation.target_status is not None
        invocation.target_status.write_text("{}", encoding="utf-8")
    _install_completed_process(
        monkeypatch,
        _CompletedProcess(0),
        cleanup=profile_server.ProcessGroupCleanup(
            existed=True,
            complete=True,
            kill_required=False,
        ),
    )
    times = iter((10.0, 10.0 + elapsed))
    monkeypatch.setattr(profile_server.time, "monotonic", lambda: next(times))

    result = profile_server._run_invocation(
        invocation,
        profiler="py-spy",
        duration=5,
        delay=0,
    )

    assert result == expected


def test_py_spy_target_binding_accepts_live_identity_with_samples(
    monkeypatch, tmp_path
) -> None:
    invocation = profile_server.build_invocation(
        "py-spy",
        executable="/tools/py-spy",
        output=tmp_path / "server.speedscope.json",
        duration=5,
        serve_args=("--", "deploy/gpu.yaml"),
    )
    _record_live_pyspy_target(monkeypatch, invocation)

    assert profile_server._pyspy_target_is_live(
        invocation, process_group=_CompletedProcess.pid
    )


@pytest.mark.parametrize(
    "failure",
    ["missing-start", "zombie", "wrong-group", "reused-pid", "no-samples"],
)
def test_py_spy_target_binding_rejects_wrong_or_unsampled_process(
    failure, monkeypatch, tmp_path
) -> None:
    invocation = profile_server.build_invocation(
        "py-spy",
        executable="/tools/py-spy",
        output=tmp_path / "server.speedscope.json",
        duration=5,
        serve_args=("--", "deploy/gpu.yaml"),
    )
    recorded = _record_live_pyspy_target(
        monkeypatch,
        invocation,
        samples=failure != "no-samples",
    )
    if failure == "missing-start":
        assert invocation.target_started is not None
        invocation.target_started.unlink()
    elif failure != "no-samples":
        current = profile_server.ProcessIdentity(
            pid=recorded.pid,
            process_group=(
                recorded.process_group + 1
                if failure == "wrong-group"
                else recorded.process_group
            ),
            state="Z" if failure == "zombie" else "S",
            start_time_ticks=(
                recorded.start_time_ticks + 1
                if failure == "reused-pid"
                else recorded.start_time_ticks
            ),
        )
        monkeypatch.setattr(
            profile_server,
            "_read_process_identity",
            lambda _pid: current,
        )

    assert not profile_server._pyspy_target_is_live(
        invocation, process_group=_CompletedProcess.pid
    )


def test_py_spy_zero_exit_requires_a_residual_supervised_server_group(
    monkeypatch, tmp_path
) -> None:
    output = tmp_path / "server.speedscope.json"
    invocation = profile_server.build_invocation(
        "py-spy",
        executable="/tools/py-spy",
        output=output,
        duration=5,
        serve_args=("--", "deploy/gpu.yaml"),
    )
    _record_live_pyspy_target(monkeypatch, invocation)
    _install_completed_process(monkeypatch, _CompletedProcess(0))
    times = iter((10.0, 15.0))
    monkeypatch.setattr(profile_server.time, "monotonic", lambda: next(times))

    result = profile_server._run_invocation(
        invocation,
        profiler="py-spy",
        duration=5,
        delay=0,
    )

    assert result == profile_server.INVALID_CAPTURE_EXIT


def test_sigkill_required_for_residual_group_makes_capture_nonzero(
    monkeypatch, tmp_path
) -> None:
    output = tmp_path / "server.speedscope.json"
    invocation = profile_server.build_invocation(
        "py-spy",
        executable="/tools/py-spy",
        output=output,
        duration=5,
        serve_args=("--", "deploy/gpu.yaml"),
    )
    _record_live_pyspy_target(monkeypatch, invocation)
    _install_completed_process(
        monkeypatch,
        _CompletedProcess(0),
        cleanup=profile_server.ProcessGroupCleanup(
            existed=True,
            complete=True,
            kill_required=True,
        ),
    )
    times = iter((10.0, 15.0))
    monkeypatch.setattr(profile_server.time, "monotonic", lambda: next(times))

    result = profile_server._run_invocation(
        invocation,
        profiler="py-spy",
        duration=5,
        delay=0,
    )

    assert result == profile_server.TIMEOUT_EXIT


@pytest.mark.parametrize("profiler", ["nsys", "py-spy"])
def test_other_profiler_failures_are_preserved(profiler, monkeypatch, tmp_path) -> None:
    output = tmp_path / (
        "server.nsys-rep" if profiler == "nsys" else "server.speedscope.json"
    )
    invocation = profile_server.build_invocation(
        profiler,
        executable=f"/tools/{profiler}",
        output=output,
        duration=5,
        serve_args=("--", "deploy/gpu.yaml"),
    )
    _install_completed_process(monkeypatch, _CompletedProcess(23))
    times = iter((10.0, 15.0))
    monkeypatch.setattr(profile_server.time, "monotonic", lambda: next(times))

    assert (
        profile_server._run_invocation(
            invocation,
            profiler=profiler,
            duration=5,
            delay=0,
        )
        == 23
    )


@pytest.mark.parametrize(
    ("received_signal", "forwarded_signal"),
    [(signal.SIGTERM, signal.SIGTERM)]
    + ([(signal.SIGHUP, signal.SIGTERM)] if hasattr(signal, "SIGHUP") else []),
    ids=("sigterm", "sighup") if hasattr(signal, "SIGHUP") else ("sigterm",),
)
def test_external_signal_is_forwarded_to_the_private_process_group(
    received_signal, forwarded_signal, monkeypatch, tmp_path
) -> None:
    output = tmp_path / "server.speedscope.json"
    invocation = profile_server.build_invocation(
        "py-spy",
        executable="/tools/py-spy",
        output=output,
        duration=5,
        serve_args=("--", "deploy/gpu.yaml"),
    )
    installed_handlers: dict[int, object] = {}
    restored_handlers: list[tuple[int, object]] = []
    previous_handler = object()

    monkeypatch.setattr(
        profile_server.signal,
        "getsignal",
        lambda _signum: previous_handler,
    )

    def fake_signal(signum, handler):
        if handler is previous_handler:
            restored_handlers.append((signum, handler))
        else:
            installed_handlers[signum] = handler

    monkeypatch.setattr(profile_server.signal, "signal", fake_signal)
    forwarded: list[tuple[int, int]] = []
    monkeypatch.setattr(
        profile_server.os,
        "killpg",
        lambda pid, signum: forwarded.append((pid, signum)),
    )

    def send_signal() -> None:
        handler = installed_handlers[received_signal]
        assert callable(handler)
        handler(received_signal, None)

    _install_completed_process(
        monkeypatch,
        _CompletedProcess(-forwarded_signal, on_wait=send_signal),
    )
    times = iter((10.0, 11.0))
    monkeypatch.setattr(profile_server.time, "monotonic", lambda: next(times))

    result = profile_server._run_invocation(
        invocation,
        profiler="py-spy",
        duration=5,
        delay=0,
    )

    assert result == 128 + received_signal
    assert forwarded == [(123, forwarded_signal)]
    expected_signals = {signal.SIGINT, signal.SIGTERM}
    if hasattr(signal, "SIGHUP"):
        expected_signals.add(signal.SIGHUP)
    assert {signum for signum, _handler in restored_handlers} == expected_signals


def test_external_deadline_terminates_then_kills_a_stuck_profiler(
    monkeypatch, tmp_path
) -> None:
    output = tmp_path / "server.speedscope.json"
    invocation = profile_server.build_invocation(
        "py-spy",
        executable="/tools/py-spy",
        output=output,
        duration=5,
        serve_args=("--", "deploy/gpu.yaml"),
    )

    class StuckProcess:
        pid = 456

        @staticmethod
        def poll():
            return None

        @staticmethod
        def wait(timeout=None):
            raise subprocess.TimeoutExpired("py-spy", timeout)

    _install_completed_process(monkeypatch, StuckProcess())
    forwarded: list[tuple[int, int]] = []
    monkeypatch.setattr(
        profile_server.os,
        "killpg",
        lambda pid, signum: forwarded.append((pid, signum)),
    )
    hard_deadline = 5.0 + profile_server.REPORT_FINALIZE_GRACE_SECONDS
    times = iter(
        (
            0.0,
            hard_deadline,
            hard_deadline + profile_server.CANCEL_GRACE_SECONDS,
            hard_deadline
            + profile_server.CANCEL_GRACE_SECONDS
            + profile_server.KILL_WAIT_SECONDS,
        )
    )
    monkeypatch.setattr(profile_server.time, "monotonic", lambda: next(times))

    result = profile_server._run_invocation(
        invocation,
        profiler="py-spy",
        duration=5,
        delay=0,
    )

    assert result == profile_server.TIMEOUT_EXIT
    assert forwarded == [(456, signal.SIGTERM), (456, signal.SIGKILL)]


def test_script_help_is_available_without_profiler_dependencies() -> None:
    result = os.spawnv(
        os.P_WAIT,
        sys.executable,
        [sys.executable, "scripts/profile_server.py", "--help"],
    )

    assert result == 0
