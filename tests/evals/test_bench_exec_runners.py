"""Pluggable local/Docker execution runner contract and hardening."""

from __future__ import annotations

import io
import stat
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

import evals.execution as execution
from evals.execution import (
    DockerExecutionRunner,
    ExecResult,
    ExecutionRunner,
    LocalExecutionRunner,
    build_execution_runner,
)

_DIGEST = "a" * 64
_IMAGE = f"registry.example/kairyu/execution@sha256:{_DIGEST}"
_CONTAINER_ID = "c" * 64


class FakeProcess:
    def __init__(
        self,
        *,
        stdout: bytes = b"",
        stderr: bytes = execution._DOCKER_READY_MARKER.encode(),
        returncode: int = 0,
        times_out: bool = False,
    ) -> None:
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self.returncode = returncode
        self.pid = 987_654_321
        self.times_out = times_out
        self.wait_calls = 0
        self.killed = False

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self.times_out and self.wait_calls == 1:
            raise subprocess.TimeoutExpired(["docker", "start"], timeout)
        return self.returncode

    def kill(self) -> None:
        self.killed = True


class WaitErrorProcess(FakeProcess):
    def wait(self, timeout=None):
        self.wait_calls += 1
        if self.wait_calls == 1:
            raise OSError("wait interrupted")
        return self.returncode


class CommandRecorder:
    def __init__(
        self,
        *,
        daemon_ok: bool = True,
        image_ok: bool = True,
        removal_ok: bool = True,
    ) -> None:
        self.commands: list[list[str]] = []
        self.daemon_ok = daemon_ok
        self.image_ok = image_ok
        self.removal_ok = removal_ok
        self.container_present = False
        self.staged: dict[str, tuple[bytes, int]] = {}
        self.input_dir_mode: int | None = None

    def __call__(self, command, **kwargs):
        self.commands.append(list(command))
        expected_kwargs = {
            "capture_output": True,
            "check": False,
            "timeout": execution._CONTROL_TIMEOUT_S,
        }
        if command[:2] == ["docker", "create"]:
            expected_kwargs = {
                **expected_kwargs,
                "timeout": None,
                "start_new_session": True,
            }
        assert kwargs == expected_kwargs
        if command == ["docker", "info"]:
            return subprocess.CompletedProcess(
                command,
                0 if self.daemon_ok else 1,
                b"",
                b"" if self.daemon_ok else b"daemon refused",
            )
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(
                command,
                0 if self.image_ok else 1,
                (
                    (
                        '[{"Id":"sha256:'
                        + _DIGEST
                        + '","Os":"linux","Architecture":"amd64"}]'
                    ).encode()
                    if self.image_ok
                    else b""
                ),
                b"" if self.image_ok else b"no such image",
            )
        if command[:2] == ["docker", "create"]:
            mount = command[command.index("--mount") + 1]
            source = next(
                field.removeprefix("src=")
                for field in mount.split(",")
                if field.startswith("src=")
            )
            input_dir = execution.Path(source)
            self.input_dir_mode = stat.S_IMODE(input_dir.stat().st_mode)
            self.staged = {
                path.name: (path.read_bytes(), stat.S_IMODE(path.stat().st_mode))
                for path in input_dir.iterdir()
            }
            self.container_present = True
            return subprocess.CompletedProcess(command, 0, (_CONTAINER_ID + "\n").encode(), b"")
        if command[:3] == ["docker", "rm", "-f"]:
            if self.removal_ok:
                self.container_present = False
            return subprocess.CompletedProcess(
                command,
                0 if self.removal_ok else 1,
                b"",
                b"" if self.removal_ok else b"remove failed",
            )
        if command[:3] == ["docker", "ps", "-aq"]:
            return subprocess.CompletedProcess(
                command,
                0,
                (_CONTAINER_ID + "\n").encode() if self.container_present else b"",
                b"",
            )
        return subprocess.CompletedProcess(command, 0, b"", b"")


class PopenRecorder:
    def __init__(self, process: FakeProcess) -> None:
        self.process = process
        self.command: list[str] | None = None
        self.kwargs: dict | None = None

    def __call__(self, command, **kwargs):
        self.command = list(command)
        self.kwargs = kwargs
        return self.process


class DelayedCreateRecorder(CommandRecorder):
    def __init__(self) -> None:
        super().__init__()
        self.create_started = threading.Event()
        self.release_create = threading.Event()
        self.removed = threading.Event()

    def __call__(self, command, **kwargs):
        if command[:2] == ["docker", "create"]:
            self.create_started.set()
            result = super().__call__(command, **kwargs)
            assert self.release_create.wait(timeout=2)
        else:
            result = super().__call__(command, **kwargs)
        if command[:3] == ["docker", "rm", "-f"]:
            self.removed.set()
        return result


def _docker_runner(
    process: FakeProcess | None = None,
    *,
    control: CommandRecorder | None = None,
    name: str = "kairyu-exec-fixed",
) -> tuple[DockerExecutionRunner, PopenRecorder, CommandRecorder]:
    fake_process = process or FakeProcess()
    popen = PopenRecorder(fake_process)
    commands = control or CommandRecorder()
    runner = DockerExecutionRunner(
        _IMAGE,
        cpus=1.5,
        pids_limit=17,
        disk_mb=321,
        _popen_factory=popen,
        _run_command=commands,
        _which=lambda _name: "/usr/bin/docker",
        _name_factory=lambda: name,
        _sleep=lambda _seconds: None,
    )
    return runner, popen, commands


def test_runner_protocol_and_local_metadata_availability():
    runner = LocalExecutionRunner()
    assert isinstance(runner, ExecutionRunner)
    assert runner.metadata()["runner"] == "local"
    assert runner.metadata()["security_boundary"] is False
    assert runner.metadata()["numeric_library_threads"] == 1
    assert runner.availability()[0] is True


def test_local_runner_preserves_execution_contract_and_caps_output():
    runner = LocalExecutionRunner()
    result = runner.run_python(
        "import os, sys\n"
        "print(open('payload.txt').read())\n"
        "print(sys.stdin.read().upper())\n"
        "print('x' * 70000)\n"
        "print('HF_TOKEN' in os.environ)\n",
        stdin="hello",
        files={"payload.txt": b"payload"},
    )
    assert result.ok
    assert result.stdout.startswith("payload\nHELLO\n")
    assert result.stdout.endswith("x")
    assert len(result.stdout.encode()) == 64_000
    assert "True" not in result.stdout


@pytest.mark.parametrize(
    "name",
    [
        "",
        ".",
        "..",
        "main.py",
        "stdin",
        "../escape",
        "nested/file",
        r"nested\file",
        "C:drive",
        "nul\x00byte",
    ],
)
def test_all_runners_reject_unsafe_or_reserved_file_names(name):
    with pytest.raises(ValueError, match="file name"):
        LocalExecutionRunner().run_python("pass", files={name: b"x"})


def test_execution_values_and_module_names_are_validated_before_launch():
    runner = LocalExecutionRunner()
    with pytest.raises(ValueError, match="timeout_s"):
        runner.run_python("pass", timeout_s=float("nan"))
    with pytest.raises(ValueError, match="memory_mb"):
        runner.run_python("pass", memory_mb=True)
    with pytest.raises(TypeError, match="contain bytes"):
        runner.run_python("pass", files={"x": "not bytes"})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="module name"):
        runner.has_module("os; raise RuntimeError()")


@pytest.mark.parametrize(
    "image",
    [
        "python:3.12",
        "sha256:" + "A" * 64,
        "sha256:" + "a" * 63,
        "repo@sha256:" + "a" * 63,
        "-repo@sha256:" + "a" * 64,
        "--pull=always@sha256:" + "a" * 64,
        "repo@@sha256:" + "a" * 64,
    ],
)
def test_docker_runner_rejects_mutable_or_option_like_images(image):
    with pytest.raises(ValueError, match="image"):
        DockerExecutionRunner(image)


def test_docker_runner_accepts_only_content_addressed_image_forms():
    assert DockerExecutionRunner(f"sha256:{_DIGEST}").image == f"sha256:{_DIGEST}"
    assert DockerExecutionRunner(_IMAGE).image == _IMAGE


def test_build_execution_runner_uses_exact_config_and_rejects_ambiguity():
    assert isinstance(build_execution_runner({"runner": "local"}), LocalExecutionRunner)
    with pytest.raises(ValueError, match="does not accept an image"):
        build_execution_runner({"runner": "local", "image": _IMAGE})
    runner = build_execution_runner(
        SimpleNamespace(
            runner="docker",
            image=_IMAGE,
            cpus=2.0,
            pids_limit=32,
            disk_mb=512,
            pull_policy="never",
        )
    )
    assert isinstance(runner, DockerExecutionRunner)
    assert runner.metadata() | {} == {
        "runner": "docker",
        "image": _IMAGE,
        "pull_policy": "never",
        "log_driver": "none",
        "restart_policy": "no",
        "cleanup_label": "io.evals-exec=true",
        "network": "none",
        "rootfs": "read-only",
        "user": "65534:65534",
        "capabilities": "none",
        "no_new_privileges": True,
        "cpus": 2.0,
        "pids_limit": 32,
        "disk_mb": 512,
        "writable_tmpfs_mb": 512,
        "shm_mb": 64,
        "input_mount": "read-only",
        "output_cap_bytes": 64_000,
        "numeric_library_threads": 1,
        "create_timeout_s": execution._CONTROL_TIMEOUT_S,
    }


def test_docker_availability_fails_closed_for_binary_daemon_and_image():
    missing = DockerExecutionRunner(_IMAGE, _which=lambda _name: None)
    assert missing.availability() == (False, "docker unavailable (binary not found)")

    daemon = CommandRecorder(daemon_ok=False)
    runner = DockerExecutionRunner(
        _IMAGE,
        _run_command=daemon,
        _which=lambda _name: "/usr/bin/docker",
    )
    available, detail = runner.availability()
    assert not available and "daemon not reachable" in detail

    image = CommandRecorder(image_ok=False)
    runner = DockerExecutionRunner(
        _IMAGE,
        _run_command=image,
        _which=lambda _name: "/usr/bin/docker",
    )
    available, detail = runner.availability()
    assert not available and "--pull=never" in detail


def test_docker_availability_binds_resolved_image_and_platform_identity():
    recorder = CommandRecorder()
    runner = DockerExecutionRunner(
        _IMAGE,
        _run_command=recorder,
        _which=lambda _name: "/usr/bin/docker",
    )

    assert runner.availability()[0] is True
    assert {
        "resolved_image_id": f"sha256:{_DIGEST}",
        "image_os": "linux",
        "image_architecture": "amd64",
        "image_variant": None,
    }.items() <= runner.metadata().items()


def test_docker_availability_rejects_missing_resolved_platform_identity():
    recorder = CommandRecorder()

    def malformed_inspect(command, **kwargs):
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, b"[]", b"")
        return recorder(command, **kwargs)

    runner = DockerExecutionRunner(
        _IMAGE,
        _run_command=malformed_inspect,
        _which=lambda _name: "/usr/bin/docker",
    )

    available, detail = runner.availability()
    assert available is False
    assert "identity unavailable" in detail


def test_docker_runner_builds_hardened_argv_and_read_only_input_bundle():
    runner, popen, commands = _docker_runner(
        FakeProcess(stdout=b"answer\n"),
    )
    hostile = "`touch /tmp/pwned`; $(id); ' \" ; --pull=always"
    result = runner.run_python(
        f"print({hostile!r})",
        stdin="input; $(id)",
        memory_mb=777,
        files={"test_data.h5": b"large-asset-stands-in-place"},
    )
    assert result == ExecResult(0, "answer\n", "", False)
    assert commands.commands[:2] == [
        ["docker", "info"],
        ["docker", "image", "inspect", _IMAGE],
    ]
    command = next(command for command in commands.commands if command[:2] == ["docker", "create"])
    assert hostile not in command
    assert "input; $(id)" not in command
    assert command[command.index("--name") + 1] == "kairyu-exec-fixed"
    assert command[command.index("--pull") + 1] == "never"
    assert command[command.index("--log-driver") + 1] == "none"
    assert command[command.index("--restart") + 1] == "no"
    assert command[command.index("--label") + 1] == "io.evals-exec=true"
    for flag, expected in (
        ("--network", "none"),
        ("--cap-drop", "ALL"),
        ("--security-opt", "no-new-privileges"),
        ("--user", "65534:65534"),
        ("--cpus", "1.5"),
        ("--memory", "777m"),
        ("--memory-swap", "777m"),
        ("--pids-limit", "17"),
        ("--shm-size", "64m"),
        ("--entrypoint", "/usr/bin/env"),
    ):
        assert command[command.index(flag) + 1] == expected
    assert "--rm" in command
    assert "--init" in command
    assert "--read-only" in command
    tmpfs = command[command.index("--tmpfs") + 1]
    assert "size=321m" in tmpfs
    assert "nosuid" in tmpfs and "nodev" in tmpfs and "noexec" in tmpfs
    mount = command[command.index("--mount") + 1]
    assert mount.endswith(",dst=/input,readonly")
    assert command[command.index(_IMAGE) + 1 :][:2] == ["-i", "HOME=/work"]
    for thread_variable in (
        "OPENBLAS_NUM_THREADS=1",
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1",
        "BLIS_NUM_THREADS=1",
        "VECLIB_MAXIMUM_THREADS=1",
    ):
        assert thread_variable in command
    assert "/bin/sh" not in command and "bash" not in command
    assert "os.symlink" in command[-1] and "copy" not in command[-1]
    assert commands.input_dir_mode == 0o755
    assert commands.staged == {
        "main.py": (f"print({hostile!r})".encode(), 0o444),
        "stdin": (b"input; $(id)", 0o444),
        "test_data.h5": (b"large-asset-stands-in-place", 0o444),
    }
    assert popen.command == ["docker", "start", "--attach", _CONTAINER_ID]
    assert popen.kwargs == {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "start_new_session": True,
    }


def test_docker_availability_success_is_cached_across_executions():
    runner, _, commands = _docker_runner()
    assert runner.availability()[0]
    runner.run_python("pass")
    assert commands.commands.count(["docker", "info"]) == 1
    assert commands.commands.count(["docker", "image", "inspect", _IMAGE]) == 1


def test_docker_output_is_capped_after_internal_marker_is_removed():
    runner, _, _ = _docker_runner(FakeProcess(stdout=b"x" * 70_000))
    result = runner.run_python("pass")
    assert result.ok
    assert len(result.stdout) == 64_000
    assert result.stderr == ""


def test_docker_failure_before_bootstrap_is_infrastructure_not_wrong_answer():
    runner, _, _ = _docker_runner(
        FakeProcess(stderr=b"docker: daemon failed\n", returncode=125)
    )
    with pytest.raises(RuntimeError, match="before the isolated Python program started"):
        runner.run_python("print('correct')")


def test_docker_timeout_kills_and_removes_exact_container_then_returns_timeout():
    process = FakeProcess(times_out=True)
    runner, _, commands = _docker_runner(process)
    result = runner.run_python("import time; time.sleep(60)", timeout_s=0.01)
    assert result.timed_out and not result.ok and result.returncode == -1
    assert ["docker", "rm", "-f", _CONTAINER_ID] in commands.commands
    ps = next(command for command in commands.commands if command[:3] == ["docker", "ps", "-aq"])
    assert ps[-1] == f"id={_CONTAINER_ID}"
    assert process.killed


def test_docker_timeout_cleanup_failure_is_infrastructure_error():
    commands = CommandRecorder(removal_ok=False)
    runner, _, _ = _docker_runner(
        FakeProcess(times_out=True),
        control=commands,
    )
    with pytest.raises(RuntimeError, match="container .* remains"):
        runner.run_python("import time; time.sleep(60)", timeout_s=0.01)
    assert sum(
        command[:3] == ["docker", "rm", "-f"] for command in commands.commands
    ) >= 3


def test_docker_wait_error_still_removes_the_exact_container():
    process = WaitErrorProcess()
    runner, _, commands = _docker_runner(process)

    with pytest.raises(OSError, match="wait interrupted"):
        runner.run_python("pass")

    assert ["docker", "rm", "-f", _CONTAINER_ID] in commands.commands
    assert process.killed


def test_docker_invalid_generated_container_name_never_launches():
    runner, popen, _ = _docker_runner(name="--privileged")
    with pytest.raises(RuntimeError, match="unsafe name"):
        runner.run_python("pass")
    assert popen.command is None


def test_docker_start_failure_still_removes_the_created_container():
    commands = CommandRecorder()

    def fail_to_start(_command, **_kwargs):
        raise OSError("start failed")

    runner = DockerExecutionRunner(
        _IMAGE,
        _popen_factory=fail_to_start,
        _run_command=commands,
        _which=lambda _name: "/usr/bin/docker",
        _name_factory=lambda: "kairyu-exec-fixed",
        _sleep=lambda _seconds: None,
    )

    with pytest.raises(OSError, match="start failed"):
        runner.run_python("pass")

    assert ["docker", "rm", "-f", _CONTAINER_ID] in commands.commands
    assert not commands.container_present


def test_slow_create_times_out_caller_and_helper_removes_late_container():
    commands = DelayedCreateRecorder()
    process = FakeProcess()
    popen = PopenRecorder(process)
    runner = DockerExecutionRunner(
        _IMAGE,
        _popen_factory=popen,
        _run_command=commands,
        _which=lambda _name: "/usr/bin/docker",
        _name_factory=lambda: "kairyu-exec-fixed",
        _sleep=lambda _seconds: None,
        _create_timeout_s=0.02,
    )

    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="creation exceeded"):
            runner.run_python("pass")
    finally:
        commands.release_create.set()

    assert time.monotonic() - started < 1.0
    assert commands.create_started.is_set()
    assert popen.command is None
    assert commands.removed.wait(timeout=1), commands.commands
    assert not commands.container_present
