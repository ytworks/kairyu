"""Pluggable Python execution for code-scored benchmarks.

``LocalExecutionRunner`` preserves the historical trusted-development runner.
``DockerExecutionRunner`` is the untrusted-code boundary: it runs one
content-addressed image with no network, a read-only root, no capabilities, and
bounded cgroup/tmpfs resources.  Both return the same small ``ExecResult``.

The Docker path deliberately talks to the Docker CLI with an argv list.  Model
code, stdin, filenames, image references, and paths are never interpolated into
a shell command.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Protocol, runtime_checkable

_OUTPUT_CAP = 64_000
_FILE_SIZE_LIMIT = 32_000_000
_CONTROL_TIMEOUT_S = 10.0
_CONTAINER_USER = "65534:65534"
_CONTAINER_LABEL = "io.evals-exec=true"
_RESERVED_INPUT_NAMES = frozenset({"main.py", "stdin"})
_IMAGE_ID_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_REPOSITORY_DIGEST_RE = re.compile(r"[^@\s]+@sha256:[0-9a-f]{64}\Z")
_CONTAINER_ID_RE = re.compile(r"[0-9a-f]{64}\Z")
_MODULE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\Z")
_DOCKER_READY_MARKER = "__KAIRYU_EXEC_STARTED_8c5a60f4__\n"
_NUMERIC_THREAD_ENV = (
    ("OPENBLAS_NUM_THREADS", "1"),
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("BLIS_NUM_THREADS", "1"),
    ("VECLIB_MAXIMUM_THREADS", "1"),
)

# Constant code only.  User-controlled values live in the read-only /input
# mount, never in this argument.  Large read-only fixture inputs
# stay in that mount; symlinks expose them beside main.py without copying them
# into the memory-backed /work quota.
_DOCKER_BOOTSTRAP = """\
import os
import pathlib
import sys

source_root = pathlib.Path("/input")
work_root = pathlib.Path("/work")
for source in source_root.iterdir():
    if source.name != "stdin":
        os.symlink(source, work_root / source.name)
stdin_fd = os.open(source_root / "stdin", os.O_RDONLY)
os.dup2(stdin_fd, 0)
os.close(stdin_fd)
os.chdir(work_root)
os.write(2, b"__KAIRYU_EXEC_STARTED_8c5a60f4__\\n")
os.execv(sys.executable, [sys.executable, "-I", "main.py"])
"""


@dataclass(frozen=True)
class ExecResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


@runtime_checkable
class ExecutionRunner(Protocol):
    """One explicit execution environment used by benchmark adapters."""

    def run_python(
        self,
        code: str,
        *,
        stdin: str = "",
        timeout_s: float = 30.0,
        memory_mb: int = 4096,
        files: Mapping[str, bytes] | None = None,
    ) -> ExecResult: ...

    def has_module(self, name: str) -> bool: ...

    def metadata(self) -> dict[str, object]: ...

    def availability(self) -> tuple[bool, str]: ...


def _validate_execution(
    code: str,
    stdin: str,
    timeout_s: float,
    memory_mb: int,
    files: Mapping[str, bytes] | None,
) -> dict[str, bytes]:
    if not isinstance(code, str):
        raise TypeError("code must be a string")
    if not isinstance(stdin, str):
        raise TypeError("stdin must be a string")
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(timeout_s)
        or timeout_s <= 0
    ):
        raise ValueError("timeout_s must be finite and positive")
    if isinstance(memory_mb, bool) or not isinstance(memory_mb, int) or memory_mb < 1:
        raise ValueError("memory_mb must be a positive integer")
    staged: dict[str, bytes] = {}
    for name, data in (files or {}).items():
        if not isinstance(name, str) or not name:
            raise ValueError("execution file names must be non-empty strings")
        if (
            name in _RESERVED_INPUT_NAMES
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
            or "\x00" in name
            or PureWindowsPath(name).drive
        ):
            raise ValueError(
                f"unsafe execution file name {name!r}: expected one non-reserved "
                "path component"
            )
        if not isinstance(data, bytes):
            raise TypeError(f"execution file {name!r} must contain bytes")
        staged[name] = data
    return staged


def _validate_module_name(name: str) -> str:
    if not isinstance(name, str) or _MODULE_RE.fullmatch(name) is None:
        raise ValueError(f"invalid Python module name {name!r}")
    return name


def _stage_inputs(
    root: Path,
    *,
    code: str,
    stdin: str,
    files: Mapping[str, bytes],
) -> None:
    """Write one read-only bundle that uid 65534 can traverse and read."""

    root.chmod(0o755)
    payloads = {
        "main.py": code.encode("utf-8"),
        "stdin": stdin.encode("utf-8"),
        **files,
    }
    for name, data in payloads.items():
        path = root / name
        path.write_bytes(data)
        path.chmod(0o444)


def _make_preexec(memory_mb: int, cpu_s: int):
    import resource

    def preexec() -> None:  # pragma: no cover - runs in the forked child
        os.setsid()
        limit = memory_mb * 1024 * 1024
        for memory_rlimit in (resource.RLIMIT_AS, resource.RLIMIT_DATA):
            try:
                resource.setrlimit(memory_rlimit, (limit, limit))
                break
            except (ValueError, OSError):
                continue
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(
            resource.RLIMIT_FSIZE,
            (_FILE_SIZE_LIMIT, _FILE_SIZE_LIMIT),
        )

    return preexec


def _kill_process_group(process) -> None:
    try:
        # Both launch paths create a new session, so the group ID is the
        # original process PID.  Use it directly: after the group leader exits
        # and is reaped, getpgid(pid) can no longer discover still-live
        # descendants in that group.
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            process.kill()
        except (ProcessLookupError, OSError):
            pass


class _CappedPipe:
    """Drain a pipe completely while retaining at most the public output cap."""

    def __init__(self, pipe, cap: int = _OUTPUT_CAP) -> None:
        self._pipe = pipe
        self._cap = cap
        self._data = bytearray()
        self._thread = threading.Thread(target=self._drain, daemon=True)

    def _drain(self) -> None:
        while True:
            chunk = self._pipe.read(8192)
            if not chunk:
                return
            remaining = self._cap - len(self._data)
            if remaining > 0:
                self._data.extend(chunk[:remaining])

    def start(self) -> None:
        self._thread.start()

    def finish(self) -> bytes:
        self._thread.join(timeout=_CONTROL_TIMEOUT_S)
        if self._thread.is_alive():
            raise RuntimeError("execution output pipe did not close after process exit")
        return bytes(self._data)


def _wait_capped(
    process,
    *,
    timeout_s: float,
    timeout_cleanup: Callable[[], None],
    post_wait_cleanup: Callable[[], None] | None = None,
) -> ExecResult:
    assert process.stdout is not None
    assert process.stderr is not None
    stdout = _CappedPipe(process.stdout)
    stderr = _CappedPipe(process.stderr)
    stdout.start()
    stderr.start()
    timed_out = False
    cleanup_error: BaseException | None = None
    try:
        returncode = process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            timeout_cleanup()
        except BaseException as error:  # cleanup failure must not look like a score
            cleanup_error = error
        _kill_process_group(process)
        try:
            returncode = process.wait(timeout=_CONTROL_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except (ProcessLookupError, OSError):
                pass
            returncode = process.wait(timeout=_CONTROL_TIMEOUT_S)
    except BaseException:
        # Cancellation, an interrupted wait, or a control-plane error must not
        # bypass the same process/container cleanup used for a wall timeout.
        abort_cleanup_error: BaseException | None = None
        try:
            timeout_cleanup()
        except BaseException as error:
            abort_cleanup_error = error
        _kill_process_group(process)
        try:
            process.wait(timeout=_CONTROL_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except (ProcessLookupError, OSError):
                pass
            try:
                process.wait(timeout=_CONTROL_TIMEOUT_S)
            except BaseException as error:
                abort_cleanup_error = abort_cleanup_error or error
        try:
            stdout.finish()
            stderr.finish()
        except BaseException as error:
            abort_cleanup_error = abort_cleanup_error or error
        if abort_cleanup_error is not None:
            raise RuntimeError(
                "execution aborted and cleanup could not be verified"
            ) from abort_cleanup_error
        raise
    else:
        # A successful group leader may have left descendants behind. Reap
        # them before joining inherited stdout/stderr pipes.
        if post_wait_cleanup is not None:
            post_wait_cleanup()
    stdout_bytes = stdout.finish()
    stderr_bytes = stderr.finish()
    if cleanup_error is not None:
        raise RuntimeError(
            "execution timed out and cleanup could not be verified"
        ) from cleanup_error
    return ExecResult(
        returncode=-1 if timed_out else int(returncode),
        stdout=stdout_bytes.decode(errors="replace"),
        stderr=stderr_bytes.decode(errors="replace"),
        timed_out=timed_out,
    )


class LocalExecutionRunner:
    """The historical local subprocess runner for trusted development."""

    def __init__(
        self,
        *,
        _popen_factory: Callable[..., object] = subprocess.Popen,
    ) -> None:
        self._popen_factory = _popen_factory

    def run_python(
        self,
        code: str,
        *,
        stdin: str = "",
        timeout_s: float = 30.0,
        memory_mb: int = 4096,
        files: Mapping[str, bytes] | None = None,
    ) -> ExecResult:
        staged_files = _validate_execution(
            code,
            stdin,
            timeout_s,
            memory_mb,
            files,
        )
        with tempfile.TemporaryDirectory(prefix="kairyu-bench-") as tmp:
            workdir = Path(tmp)
            _stage_inputs(
                workdir,
                code=code,
                stdin=stdin,
                files=staged_files,
            )
            env = {
                "PATH": "/usr/bin:/bin",
                "HOME": tmp,
                "TMPDIR": tmp,
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                **dict(_NUMERIC_THREAD_ENV),
            }
            with (workdir / "stdin").open("rb") as input_stream:
                process = self._popen_factory(
                    [sys.executable, "-I", str(workdir / "main.py")],
                    stdin=input_stream,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=tmp,
                    env=env,
                    preexec_fn=_make_preexec(memory_mb, int(timeout_s) + 1),
                )
                return _wait_capped(
                    process,
                    timeout_s=float(timeout_s),
                    timeout_cleanup=lambda: _kill_process_group(process),
                    post_wait_cleanup=lambda: _kill_process_group(process),
                )

    def has_module(self, name: str) -> bool:
        module = _validate_module_name(name)
        result = self.run_python(
            f"import importlib\nimportlib.import_module({module!r})\n",
            timeout_s=20.0,
        )
        return result.ok

    def metadata(self) -> dict[str, object]:
        return {
            "runner": "local",
            "python": sys.executable,
            "isolation": "subprocess",
            "security_boundary": False,
            "output_cap_bytes": _OUTPUT_CAP,
            "numeric_library_threads": 1,
        }

    def availability(self) -> tuple[bool, str]:
        return True, f"local Python available ({sys.executable})"


@dataclass
class _CreateState:
    lock: object = field(default_factory=threading.Lock)
    finished: threading.Event = field(default_factory=threading.Event)
    ownership: threading.Event = field(default_factory=threading.Event)
    container_id: str | None = None
    error: BaseException | None = None
    abandoned: bool = False
    claimed: bool = False


class _ContainerLease:
    """Cleanup ownership armed before Docker create can return an ID."""

    def __init__(self, remove: Callable[[str], None]) -> None:
        self._remove = remove
        self.container_id: str | None = None

    def cleanup(self) -> None:
        container_id, self.container_id = self.container_id, None
        if container_id is not None:
            self._remove(container_id)


class DockerExecutionRunner:
    """One hardened, content-addressed Docker container per Python execution."""

    def __init__(
        self,
        image: str,
        *,
        cpus: float = 1.0,
        pids_limit: int = 64,
        disk_mb: int = 256,
        pull_policy: str = "never",
        _popen_factory: Callable[..., object] = subprocess.Popen,
        _run_command: Callable[..., object] = subprocess.run,
        _which: Callable[[str], str | None] = shutil.which,
        _name_factory: Callable[[], str] | None = None,
        _sleep: Callable[[float], None] = time.sleep,
        _create_timeout_s: float = _CONTROL_TIMEOUT_S,
    ) -> None:
        self.image = _validate_image(image)
        if (
            isinstance(cpus, bool)
            or not isinstance(cpus, (int, float))
            or not math.isfinite(cpus)
            or cpus <= 0
        ):
            raise ValueError("Docker execution cpus must be finite and positive")
        if isinstance(pids_limit, bool) or not isinstance(pids_limit, int) or pids_limit < 2:
            raise ValueError("Docker execution pids_limit must be an integer >= 2")
        if (
            isinstance(disk_mb, bool)
            or not isinstance(disk_mb, int)
            or not 1 <= disk_mb <= 65_536
        ):
            raise ValueError("Docker execution disk_mb must be in [1, 65536]")
        if pull_policy != "never":
            raise ValueError("Docker execution pull_policy must be 'never'")
        if (
            isinstance(_create_timeout_s, bool)
            or not isinstance(_create_timeout_s, (int, float))
            or not math.isfinite(_create_timeout_s)
            or _create_timeout_s <= 0
        ):
            raise ValueError("Docker create timeout must be finite and positive")
        self.cpus = float(cpus)
        self.pids_limit = pids_limit
        self.disk_mb = disk_mb
        self.pull_policy = pull_policy
        self._popen_factory = _popen_factory
        self._run_command = _run_command
        self._which = _which
        self._name_factory = _name_factory or (
            lambda: f"kairyu-bench-{uuid.uuid4().hex}"
        )
        self._sleep = _sleep
        self._create_timeout_s = float(_create_timeout_s)
        self._available: tuple[bool, str] | None = None
        self._resolved_image_identity: dict[str, str | None] | None = None

    def _control(self, command: list[str]):
        return self._run_command(
            command,
            capture_output=True,
            check=False,
            timeout=_CONTROL_TIMEOUT_S,
        )

    def availability(self) -> tuple[bool, str]:
        if self._available is not None:
            return self._available
        if self._which("docker") is None:
            return False, "docker unavailable (binary not found)"
        try:
            info = self._control(["docker", "info"])
        except (OSError, subprocess.TimeoutExpired) as error:
            return False, f"docker unavailable ({error})"
        if info.returncode != 0:
            detail = (info.stderr or info.stdout or b"").decode(
                errors="replace"
            ).strip()
            return False, f"docker unavailable (daemon not reachable: {detail[:120]})"
        try:
            image = self._control(["docker", "image", "inspect", self.image])
        except (OSError, subprocess.TimeoutExpired) as error:
            return False, f"docker execution image unavailable ({error})"
        if image.returncode != 0:
            detail = (image.stderr or image.stdout or b"").decode(
                errors="replace"
            ).strip()
            return False, (
                f"docker execution image {self.image!r} unavailable "
                f"(--pull=never): {detail[:120]}"
            )
        try:
            inspected = json.loads((image.stdout or b"").decode("utf-8"))
            if not isinstance(inspected, list) or len(inspected) != 1:
                raise ValueError("inspect did not return exactly one image")
            image_info = inspected[0]
            if not isinstance(image_info, dict):
                raise ValueError("inspect image is not an object")
            resolved_id = image_info.get("Id")
            image_os = image_info.get("Os")
            architecture = image_info.get("Architecture")
            variant = image_info.get("Variant")
            if (
                not isinstance(resolved_id, str)
                or _IMAGE_ID_RE.fullmatch(resolved_id) is None
                or not isinstance(image_os, str)
                or not image_os
                or not isinstance(architecture, str)
                or not architecture
                or (variant is not None and (not isinstance(variant, str) or not variant))
            ):
                raise ValueError("inspect omitted immutable image/platform identity")
        except (UnicodeError, json.JSONDecodeError, ValueError) as error:
            return False, f"docker execution image identity unavailable ({error})"
        self._resolved_image_identity = {
            "resolved_image_id": resolved_id,
            "image_os": image_os,
            "image_architecture": architecture,
            "image_variant": variant,
        }
        self._available = (
            True,
            f"docker execution image available ({self.image})",
        )
        return self._available

    def metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "runner": "docker",
            "image": self.image,
            "pull_policy": "never",
            "log_driver": "none",
            "restart_policy": "no",
            "cleanup_label": _CONTAINER_LABEL,
            "network": "none",
            "rootfs": "read-only",
            "user": _CONTAINER_USER,
            "capabilities": "none",
            "no_new_privileges": True,
            "cpus": self.cpus,
            "pids_limit": self.pids_limit,
            "disk_mb": self.disk_mb,
            "writable_tmpfs_mb": self.disk_mb,
            "shm_mb": 64,
            "input_mount": "read-only",
            "output_cap_bytes": _OUTPUT_CAP,
            "numeric_library_threads": 1,
            "create_timeout_s": self._create_timeout_s,
        }
        if self._resolved_image_identity is not None:
            metadata.update(self._resolved_image_identity)
        return metadata

    def _create_command(
        self,
        *,
        name: str,
        input_dir: Path,
        memory_mb: int,
    ) -> list[str]:
        return [
            "docker",
            "create",
            "--name",
            name,
            "--pull",
            "never",
            "--rm",
            "--log-driver",
            "none",
            "--restart",
            "no",
            "--label",
            _CONTAINER_LABEL,
            "--init",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            _CONTAINER_USER,
            "--cpus",
            str(self.cpus),
            "--memory",
            f"{memory_mb}m",
            "--memory-swap",
            f"{memory_mb}m",
            "--pids-limit",
            str(self.pids_limit),
            "--shm-size",
            "64m",
            "--tmpfs",
            (
                "/work:rw,nosuid,nodev,noexec,"
                f"size={self.disk_mb}m,uid=65534,gid=65534,mode=0700"
            ),
            "--mount",
            f"type=bind,src={input_dir},dst=/input,readonly",
            "--entrypoint",
            "/usr/bin/env",
            self.image,
            "-i",
            "HOME=/work",
            "TMPDIR=/work",
            "PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin",
            "PYTHONNOUSERSITE=1",
            "PYTHONDONTWRITEBYTECODE=1",
            *(f"{key}={value}" for key, value in _NUMERIC_THREAD_ENV),
            "python",
            "-I",
            "-c",
            _DOCKER_BOOTSTRAP,
        ]

    def _container_is_absent(self, identifier: str) -> bool:
        if _CONTAINER_ID_RE.fullmatch(identifier) is not None:
            container_filter = f"id={identifier}"
        else:
            container_filter = f"name=^/{identifier}$"
        result = self._control(
            [
                "docker",
                "ps",
                "-aq",
                "--no-trunc",
                "--filter",
                container_filter,
            ]
        )
        return result.returncode == 0 and not (result.stdout or b"").strip()

    def _remove_container(
        self,
        identifier: str,
    ) -> None:
        for _ in range(3):
            try:
                self._control(["docker", "rm", "-f", identifier])
                if self._container_is_absent(identifier):
                    return
            except (OSError, subprocess.TimeoutExpired):
                pass
            self._sleep(0.1)
        raise RuntimeError(f"Docker execution container {identifier!r} remains")

    def _acquire_container(
        self,
        lease: _ContainerLease,
        *,
        name: str,
        input_dir: Path,
        memory_mb: int,
    ) -> None:
        command = self._create_command(
            name=name,
            input_dir=input_dir,
            memory_mb=memory_mb,
        )
        state = _CreateState()

        def create_to_completion() -> None:
            container_id: str | None = None
            error: BaseException | None = None
            try:
                created = self._run_command(
                    command,
                    capture_output=True,
                    check=False,
                    timeout=None,
                    start_new_session=True,
                )
                if created.returncode != 0:
                    detail = (created.stderr or created.stdout or b"").decode(
                        errors="replace"
                    ).strip()
                    self._remove_container(name)
                    raise RuntimeError(
                        f"Docker execution container creation failed: {detail[-500:]}"
                    )
                container_id = (created.stdout or b"").decode(
                    errors="replace"
                ).strip()
                if _CONTAINER_ID_RE.fullmatch(container_id) is None:
                    self._remove_container(name)
                    raise RuntimeError(
                        "Docker create did not return one exact 64-hex container ID"
                    )
            except BaseException as caught:
                error = caught
            with state.lock:
                state.container_id = container_id
                state.error = error
                state.finished.set()
            if container_id is None:
                return
            state.ownership.wait()
            with state.lock:
                abandoned = state.abandoned and not state.claimed
            if abandoned:
                self._remove_container(container_id)

        helper = threading.Thread(
            target=create_to_completion,
            name="kairyu-bench-docker-create",
            daemon=True,
        )
        helper.start()
        try:
            state.finished.wait(self._create_timeout_s)
            with state.lock:
                if state.container_id is not None and state.error is None:
                    # The caller's lease was created under an already-active
                    # finally block. Transfer ownership while holding the same
                    # lock the helper uses to decide whether it must clean up.
                    lease.container_id = state.container_id
                    state.claimed = True
                    state.ownership.set()
                    return
                state.abandoned = True
                state.ownership.set()
                error = state.error
            if error is not None:
                raise RuntimeError("Docker execution container creation failed") from error
            raise RuntimeError(
                "Docker execution container creation exceeded "
                f"{self._create_timeout_s:g}s; cleanup remains delegated"
            )
        except BaseException:
            with state.lock:
                if not state.claimed:
                    state.abandoned = True
                state.ownership.set()
            raise

    @contextmanager
    def _created_container(
        self,
        *,
        name: str,
        input_dir: Path,
        memory_mb: int,
    ):
        lease = _ContainerLease(self._remove_container)
        try:
            self._acquire_container(
                lease,
                name=name,
                input_dir=input_dir,
                memory_mb=memory_mb,
            )
            if lease.container_id is None:
                raise RuntimeError("Docker create completed without cleanup ownership")
            yield lease.container_id
        finally:
            lease.cleanup()

    def _cleanup_started(self, process, container_id: str) -> None:
        # `create` completed and transferred ownership before the timed attach
        # began, so cleanup owns one already-known immutable container ID.
        # There is no late-create window and no name can be reused underneath
        # this removal.
        _kill_process_group(process)
        try:
            process.wait(timeout=_CONTROL_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except (ProcessLookupError, OSError):
                pass
        self._remove_container(container_id)

    def run_python(
        self,
        code: str,
        *,
        stdin: str = "",
        timeout_s: float = 30.0,
        memory_mb: int = 4096,
        files: Mapping[str, bytes] | None = None,
    ) -> ExecResult:
        staged_files = _validate_execution(
            code,
            stdin,
            timeout_s,
            memory_mb,
            files,
        )
        available, detail = self.availability()
        if not available:
            raise RuntimeError(detail)
        with tempfile.TemporaryDirectory(prefix="kairyu-bench-input-") as tmp:
            input_dir = Path(tmp)
            _stage_inputs(
                input_dir,
                code=code,
                stdin=stdin,
                files=staged_files,
            )
            name = self._name_factory()
            if not re.fullmatch(r"kairyu-bench-[a-z0-9][a-z0-9_.-]*", name):
                raise RuntimeError("Docker execution name factory returned an unsafe name")
            with self._created_container(
                name=name,
                input_dir=input_dir,
                memory_mb=memory_mb,
            ) as container_id:
                process = self._popen_factory(
                    ["docker", "start", "--attach", container_id],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                )
                result = _wait_capped(
                    process,
                    timeout_s=float(timeout_s),
                    timeout_cleanup=lambda: self._cleanup_started(
                        process, container_id
                    ),
                )
                if _DOCKER_READY_MARKER not in result.stderr:
                    raise RuntimeError(
                        "Docker execution failed before the isolated Python "
                        f"program started (exit {result.returncode}): "
                        f"{result.stderr[-500:]}"
                    )
                return ExecResult(
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr.replace(_DOCKER_READY_MARKER, "", 1),
                    timed_out=result.timed_out,
                )

    def has_module(self, name: str) -> bool:
        module = _validate_module_name(name)
        result = self.run_python(
            f"import importlib\nimportlib.import_module({module!r})\n",
            timeout_s=20.0,
        )
        return result.ok


def _validate_image(image: str) -> str:
    if not isinstance(image, str) or not image or image.startswith("-"):
        raise ValueError(
            "Docker execution image must be a content-addressed image reference"
        )
    if _IMAGE_ID_RE.fullmatch(image) is not None:
        return image
    if _REPOSITORY_DIGEST_RE.fullmatch(image) is not None:
        repository = image.split("@", 1)[0]
        if repository.startswith("-") or repository.endswith(":"):
            raise ValueError("Docker execution image repository is invalid")
        return image
    raise ValueError(
        "Docker execution image must be sha256:<64 lowercase hex> or "
        "<repository>@sha256:<64 lowercase hex>"
    )


def _config_value(config: object, name: str, default: object = None) -> object:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def build_execution_runner(config: object) -> ExecutionRunner:
    """Build the configured runner without importing the Pydantic config type."""

    runner = _config_value(config, "runner", "local")
    if runner == "local":
        if _config_value(config, "image") is not None:
            raise ValueError("local execution runner does not accept an image")
        resource_values = {
            "cpus": (_config_value(config, "cpus", 1.0), 1.0),
            "pids_limit": (_config_value(config, "pids_limit", 64), 64),
            "disk_mb": (_config_value(config, "disk_mb", 256), 256),
            "pull_policy": (_config_value(config, "pull_policy", "never"), "never"),
        }
        changed = [
            name for name, (value, default) in resource_values.items() if value != default
        ]
        if changed:
            raise ValueError(
                "local execution runner does not accept Docker resource settings: "
                + ", ".join(changed)
            )
        return LocalExecutionRunner()
    if runner != "docker":
        raise ValueError(f"unknown execution runner {runner!r}")
    image = _config_value(config, "image")
    if not isinstance(image, str):
        raise ValueError("Docker execution runner requires an image")
    return DockerExecutionRunner(
        image,
        cpus=_config_value(config, "cpus", 1.0),
        pids_limit=_config_value(config, "pids_limit", 64),
        disk_mb=_config_value(config, "disk_mb", 256),
        pull_policy=_config_value(config, "pull_policy", "never"),
    )
