#!/usr/bin/env python
"""Launch ``kairyu serve`` under Nsight Systems or py-spy.

The helper intentionally launches the server instead of attaching to an existing
PID, avoiding the additional permission problem of locating and attaching to an
unrelated PID.  Host ptrace/seccomp policy still controls whether py-spy may read
the launched process.  Both profilers include the complete process tree, and
collections are time-bounded because an unbounded server trace can exhaust local
storage.
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import shlex
import shutil
import signal
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

NOT_RUN_EXIT = 77
# A cold 32B/70B load can consume several minutes before readiness.  Keep the
# trace finite while leaving enough room for load, warm-up, and a steady window.
MAX_DURATION_SECONDS = 3600
MAX_DELAY_SECONDS = 3600
CANCEL_GRACE_SECONDS = 30.0
REPORT_FINALIZE_GRACE_SECONDS = 300.0
KILL_WAIT_SECONDS = 5.0
# py-spy 0.4.2's integer duration timer can complete a few scheduler ticks
# before the wrapper's wall clock reaches the requested boundary.  The live
# target identity and PID-bound samples below remain mandatory, so this narrow
# allowance cannot turn an early target exit into a valid capture.
PYSPY_DURATION_TOLERANCE_SECONDS = 0.5
TIMEOUT_EXIT = 124
INVALID_CAPTURE_EXIT = 1


@dataclass(frozen=True)
class ProfileInvocation:
    executable: str
    argv: tuple[str, ...]
    artifact: Path
    reserved_outputs: tuple[Path, ...]
    collision_outputs: tuple[Path, ...]
    target_started: Path | None
    target_status: Path | None


@dataclass(frozen=True)
class ProcessGroupCleanup:
    existed: bool
    complete: bool
    kill_required: bool


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    process_group: int
    state: str
    start_time_ticks: int


def _bounded_seconds(value: str) -> int:
    try:
        seconds = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("duration must be an integer") from error
    if not 1 <= seconds <= MAX_DURATION_SECONDS:
        raise argparse.ArgumentTypeError(
            f"duration must be between 1 and {MAX_DURATION_SECONDS} seconds"
        )
    return seconds


def _bounded_delay(value: str) -> int:
    try:
        seconds = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("delay must be an integer") from error
    if not 0 <= seconds <= MAX_DELAY_SECONDS:
        raise argparse.ArgumentTypeError(
            f"delay must be between 0 and {MAX_DELAY_SECONDS} seconds"
        )
    return seconds


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    profilers = parser.add_subparsers(dest="profiler", required=True)
    for name in ("nsys", "py-spy"):
        profiler = profilers.add_parser(name)
        profiler.add_argument(
            "--output",
            required=True,
            type=Path,
            help=(
                "output artifact path (.nsys-rep is optional for nsys; the helper "
                "adds it when omitted)"
            ),
        )
        profiler.add_argument(
            "--duration",
            type=_bounded_seconds,
            required=True,
            help=(
                "collection length in seconds "
                f"(required; max: {MAX_DURATION_SECONDS})"
            ),
        )
        profiler.add_argument(
            "--dry-run",
            action="store_true",
            help="print the exact profiler argv without requiring the profiler",
        )
        if name == "nsys":
            profiler.add_argument(
                "--delay",
                type=_bounded_delay,
                default=0,
                help=(
                    "launch immediately but delay collection for model load/warm-up "
                    f"(default: 0; max: {MAX_DELAY_SECONDS})"
                ),
            )
        else:
            profiler.set_defaults(delay=0)
        profiler.add_argument(
            "serve_args",
            nargs=argparse.REMAINDER,
            metavar="-- CONFIG [SERVE_ARGS ...]",
            help="arguments passed verbatim to `kairyu serve`",
        )
    return parser


def _serve_command(serve_args: Sequence[str], *, python: str) -> tuple[str, ...]:
    forwarded = tuple(serve_args)
    if forwarded[:1] == ("--",):
        forwarded = forwarded[1:]
    if not forwarded:
        raise ValueError("a DeploymentSpec path is required after `--`")
    return (
        python,
        "-m",
        "kairyu.entrypoints.cli",
        "serve",
        *forwarded,
    )


def _write_json_exclusive(path: Path, payload: dict[str, int]) -> bool:
    """Atomically publish a small marker without replacing existing evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        return True
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_target_status(path: Path, *, returncode: int, elapsed_ns: int) -> bool:
    return _write_json_exclusive(
        path,
        {
            "schema_version": 1,
            "returncode": returncode,
            "elapsed_ns": elapsed_ns,
        },
    )


def _read_process_identity(pid: int) -> ProcessIdentity | None:
    if os.name != "posix" or type(pid) is not int or pid <= 0:
        return None
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        suffix = raw[raw.rindex(")") + 2 :].split()
        return ProcessIdentity(
            pid=pid,
            state=suffix[0],
            process_group=int(suffix[2]),
            start_time_ticks=int(suffix[19]),
        )
    except (FileNotFoundError, IndexError, OSError, ProcessLookupError, ValueError):
        return None


def _write_target_started(path: Path, identity: ProcessIdentity) -> bool:
    return _write_json_exclusive(
        path,
        {
            "schema_version": 1,
            "pid": identity.pid,
            "process_group": identity.process_group,
            "start_time_ticks": identity.start_time_ticks,
        },
    )


def _supervise_target(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--started", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("target", nargs=argparse.REMAINDER)
    args = parser.parse_args(list(argv))
    target = tuple(args.target)
    if target[:1] == ("--",):
        target = target[1:]
    if not target:
        return NOT_RUN_EXIT
    started_ns = time.monotonic_ns()
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(target)
        identity = _read_process_identity(process.pid)
        if (
            identity is None
            or identity.process_group != os.getpgrp()
            or not _write_target_started(args.started, identity)
        ):
            process.terminate()
            returncode = NOT_RUN_EXIT
        else:
            returncode = _shell_status(process.wait())
    except OSError:
        if process is not None:
            try:
                process.terminate()
            except OSError:
                pass
        returncode = NOT_RUN_EXIT
    try:
        _write_target_status(
            args.status,
            returncode=returncode,
            elapsed_ns=max(0, time.monotonic_ns() - started_ns),
        )
    except OSError:
        return NOT_RUN_EXIT
    return returncode


def build_invocation(
    profiler: str,
    *,
    executable: str,
    output: Path,
    duration: int,
    serve_args: Sequence[str],
    delay: int = 0,
    python: str = sys.executable,
) -> ProfileInvocation:
    if not 1 <= duration <= MAX_DURATION_SECONDS:
        raise ValueError(
            f"duration must be between 1 and {MAX_DURATION_SECONDS} seconds"
        )
    if not 0 <= delay <= MAX_DELAY_SECONDS:
        raise ValueError(f"delay must be between 0 and {MAX_DELAY_SECONDS} seconds")
    target = _serve_command(serve_args, python=python)
    if profiler == "nsys":
        output_text = str(output)
        if output_text.endswith(".nsys-rep"):
            output_base = output_text[: -len(".nsys-rep")]
            artifact = output
        else:
            output_base = output_text
            artifact = Path(f"{output_text}.nsys-rep")
        if not output_base:
            raise ValueError("nsys output must include a filename")
        reserved_outputs = (artifact, Path(f"{output_base}.sqlite"))
        collision_outputs = reserved_outputs
        target_started = None
        target_status = None
        profiler_args = (
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
            f"--output={output_base}",
            f"--delay={delay}",
            f"--duration={duration}",
        )
    elif profiler == "py-spy":
        if delay:
            raise ValueError("delay is supported only for nsys")
        artifact = output
        target_started = Path(f"{output}.target-started.json")
        target_status = Path(f"{output}.target-exit.json")
        reserved_outputs = (artifact, target_started)
        collision_outputs = (*reserved_outputs, target_status)
        profiler_args = (
            "record",
            "--subprocesses",
            "--threads",
            "--format=speedscope",
            f"--output={output}",
            f"--duration={duration}",
        )
        profiler_args = (*profiler_args, "--")
        target = (
            python,
            str(Path(__file__).resolve()),
            "_supervise",
            "--started",
            str(target_started),
            "--status",
            str(target_status),
            "--",
            *target,
        )
    else:
        raise ValueError(f"unsupported profiler: {profiler}")
    return ProfileInvocation(
        executable=executable,
        argv=(executable, *profiler_args, *target),
        artifact=artifact,
        reserved_outputs=reserved_outputs,
        collision_outputs=collision_outputs,
        target_started=target_started,
        target_status=target_status,
    )


def _shell_status(returncode: int) -> int:
    if returncode < 0:
        return 128 - returncode
    return returncode


def _complete_outputs(paths: Sequence[Path]) -> bool:
    try:
        return all(path.is_file() and path.stat().st_size > 0 for path in paths)
    except OSError:
        return False


def _nsys_trace_span_ns(sqlite_path: Path) -> int | None:
    try:
        uri = f"{sqlite_path.resolve().as_uri()}?mode=ro&immutable=1"
        with sqlite3.connect(uri, uri=True, timeout=1.0) as connection:
            columns = {
                str(row[1]).lower()
                for row in connection.execute(
                    "PRAGMA table_info(ANALYSIS_DETAILS)"
                ).fetchall()
            }
            if "duration" in columns:
                row = connection.execute(
                    "SELECT MAX(duration) FROM ANALYSIS_DETAILS"
                ).fetchone()
            elif {"starttime", "stoptime"} <= columns:
                row = connection.execute(
                    "SELECT MAX(stopTime - startTime) FROM ANALYSIS_DETAILS"
                ).fetchone()
            else:
                return None
        if row is None or row[0] is None:
            return None
        duration_ns = int(row[0])
    except (OSError, TypeError, ValueError, sqlite3.Error):
        return None
    return duration_ns if duration_ns >= 0 else None


def _nsys_capture_complete(
    invocation: ProfileInvocation, *, duration: int
) -> bool:
    if not _complete_outputs(invocation.reserved_outputs):
        return False
    sqlite_outputs = [
        path for path in invocation.reserved_outputs if path.suffix == ".sqlite"
    ]
    if len(sqlite_outputs) != 1:
        return False
    trace_span_ns = _nsys_trace_span_ns(sqlite_outputs[0])
    return trace_span_ns is not None and trace_span_ns >= duration * 1_000_000_000


def _load_started_identity(path: Path) -> ProcessIdentity | None:
    try:
        if not path.is_file() or not 0 < path.stat().st_size <= 4096:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            return None
        values = [
            payload.get("pid"),
            payload.get("process_group"),
            payload.get("start_time_ticks"),
        ]
        if any(type(value) is not int or value <= 0 for value in values):
            return None
        pid, process_group, start_time_ticks = values
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return ProcessIdentity(
        pid=pid,
        process_group=process_group,
        state="?",
        start_time_ticks=start_time_ticks,
    )


def _speedscope_has_process_samples(path: Path, *, pid: int) -> bool:
    needle = f'"name":"Process {pid} Thread '.encode()
    samples_key = b'"samples":['
    try:
        with path.open("rb") as stream, mmap.mmap(
            stream.fileno(), 0, access=mmap.ACCESS_READ
        ) as contents:
            position = contents.find(needle)
            while position >= 0:
                profile_end = contents.find(b"}", position)
                if profile_end < 0:
                    profile_end = len(contents)
                samples = contents.find(samples_key, position, profile_end)
                if samples >= 0:
                    first_sample = samples + len(samples_key)
                    while contents[first_sample : first_sample + 1] in {
                        b" ",
                        b"\t",
                        b"\r",
                        b"\n",
                    }:
                        first_sample += 1
                    if contents[first_sample : first_sample + 1] not in {b"", b"]"}:
                        return True
                position = contents.find(needle, position + len(needle))
    except (OSError, ValueError):
        return False
    return False


def _pyspy_target_is_live(
    invocation: ProfileInvocation, *, process_group: int
) -> bool:
    if (
        invocation.target_started is None
        or invocation.target_status is None
        or os.path.lexists(invocation.target_status)
    ):
        return False
    recorded = _load_started_identity(invocation.target_started)
    if recorded is None or recorded.process_group != process_group:
        return False
    current = _read_process_identity(recorded.pid)
    if current is None or current.state in {"Z", "X"}:
        return False
    if (
        current.process_group != recorded.process_group
        or current.start_time_ticks != recorded.start_time_ticks
    ):
        return False
    return _speedscope_has_process_samples(invocation.artifact, pid=recorded.pid)


def _signal_process_group(process: subprocess.Popen[bytes], signum: int) -> None:
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        return
    except PermissionError:
        try:
            process.send_signal(signum)
        except ProcessLookupError:
            pass


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _send_process_group(process_group: int, signum: int) -> bool:
    try:
        os.killpg(process_group, signum)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return True


def _wait_process_group_exit(process_group: int, timeout: float) -> bool:
    deadline = time.monotonic() + max(0.0, timeout)
    while _process_group_exists(process_group):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.05, remaining))
    return True


def _cleanup_process_group(process_group: int) -> ProcessGroupCleanup:
    if not _process_group_exists(process_group):
        return ProcessGroupCleanup(existed=False, complete=True, kill_required=False)
    if not _send_process_group(process_group, signal.SIGTERM):
        return ProcessGroupCleanup(existed=True, complete=False, kill_required=False)
    if _wait_process_group_exit(process_group, CANCEL_GRACE_SECONDS):
        return ProcessGroupCleanup(existed=True, complete=True, kill_required=False)
    if not _send_process_group(process_group, signal.SIGKILL):
        return ProcessGroupCleanup(existed=True, complete=False, kill_required=True)
    return ProcessGroupCleanup(
        existed=True,
        complete=_wait_process_group_exit(process_group, KILL_WAIT_SECONDS),
        kill_required=True,
    )


def _run_invocation(
    invocation: ProfileInvocation,
    *,
    profiler: str,
    duration: int,
    delay: int,
) -> int:
    started = time.monotonic()
    process: subprocess.Popen[bytes] | None = None
    received_signal: int | None = None
    cancellation_started: float | None = None
    deadline_started: float | None = None
    kill_sent: float | None = None
    leader_wait_timed_out = False

    def child_signal(signum: int) -> int:
        if hasattr(signal, "SIGHUP") and signum == signal.SIGHUP:
            return signal.SIGTERM
        return signum

    def forward_signal(signum: int, _frame: object) -> None:
        nonlocal cancellation_started, received_signal
        if received_signal is None:
            received_signal = signum
            cancellation_started = time.monotonic()
        if process is not None:
            forwarded = child_signal(signum)
            if process.poll() is None:
                _signal_process_group(process, forwarded)
            else:
                _send_process_group(process.pid, forwarded)

    handled_signals = (signal.SIGINT, signal.SIGTERM)
    if hasattr(signal, "SIGHUP"):
        handled_signals = (*handled_signals, signal.SIGHUP)
    previous_handlers = {signum: signal.getsignal(signum) for signum in handled_signals}
    for signum in handled_signals:
        signal.signal(signum, forward_signal)
    try:
        process = subprocess.Popen(invocation.argv, start_new_session=True)
        if received_signal is not None and process.poll() is None:
            _signal_process_group(process, child_signal(received_signal))
        hard_deadline = (
            started + delay + duration + REPORT_FINALIZE_GRACE_SECONDS
        )
        while True:
            try:
                returncode = process.wait(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                now = time.monotonic()
                if (
                    received_signal is None
                    and deadline_started is None
                    and now >= hard_deadline
                ):
                    deadline_started = now
                    _signal_process_group(process, signal.SIGTERM)
                termination_started = (
                    cancellation_started
                    if cancellation_started is not None
                    else deadline_started
                )
                if (
                    termination_started is not None
                    and kill_sent is None
                    and now - termination_started >= CANCEL_GRACE_SECONDS
                ):
                    _signal_process_group(process, signal.SIGKILL)
                    kill_sent = now
                if kill_sent is not None and now - kill_sent >= KILL_WAIT_SECONDS:
                    leader_wait_timed_out = True
                    returncode = process.poll()
                    if returncode is None:
                        returncode = -signal.SIGKILL
                    break
        pyspy_target_live = profiler == "py-spy" and _pyspy_target_is_live(
            invocation,
            process_group=process.pid,
        )
        cleanup = _cleanup_process_group(process.pid)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    if received_signal is not None:
        return 128 + received_signal
    if (
        deadline_started is not None
        or leader_wait_timed_out
        or not cleanup.complete
        or cleanup.kill_required
    ):
        return TIMEOUT_EXIT
    elapsed = time.monotonic() - started
    status = _shell_status(returncode)
    expected_nsys_sigterm = 128 + signal.SIGTERM
    if profiler == "nsys":
        complete = _nsys_capture_complete(invocation, duration=duration)
        if status == expected_nsys_sigterm and complete:
            return 0
        if status == 0 and not complete:
            return INVALID_CAPTURE_EXIT
    elif status == 0 and (
        elapsed + PYSPY_DURATION_TOLERANCE_SECONDS < duration
        or not _complete_outputs(invocation.reserved_outputs)
        or not cleanup.existed
        or not pyspy_target_live
    ):
        return INVALID_CAPTURE_EXIT
    return status


def _reserve_output(path: Path) -> tuple[int, int]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        return metadata.st_dev, metadata.st_ino
    finally:
        os.close(descriptor)


def _remove_empty_reservation(path: Path, identity: tuple[int, int]) -> None:
    try:
        metadata = os.stat(path, follow_symlinks=False)
        if (
            stat.S_ISREG(metadata.st_mode)
            and (metadata.st_dev, metadata.st_ino) == identity
            and metadata.st_size == 0
        ):
            path.unlink()
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    if effective_argv[:1] == ["_supervise"]:
        return _supervise_target(effective_argv[1:])
    args = _parser().parse_args(effective_argv)
    resolved_executable = shutil.which(args.profiler)
    try:
        invocation = build_invocation(
            args.profiler,
            executable=resolved_executable or args.profiler,
            output=args.output,
            duration=args.duration,
            serve_args=args.serve_args,
            delay=args.delay,
        )
    except ValueError as error:
        print(f"profiling not run: {error}", file=sys.stderr)
        return NOT_RUN_EXIT
    if args.dry_run:
        print(f"diagnostic-only profiling artifact: {invocation.artifact}")
        print(f"+ {shlex.join(invocation.argv)}")
        return 0
    if resolved_executable is None:
        print(
            f"profiling not run: required executable not found: {args.profiler}",
            file=sys.stderr,
        )
        return NOT_RUN_EXIT
    try:
        invocation.artifact.parent.mkdir(parents=True, exist_ok=True)
        for output in invocation.collision_outputs:
            if os.path.lexists(output):
                print(
                    f"profiling not run: output artifact already exists: {output}",
                    file=sys.stderr,
                )
                return NOT_RUN_EXIT
    except OSError as error:
        print(
            "profiling not run: could not prepare output directory: "
            f"{type(error).__name__}",
            file=sys.stderr,
        )
        return NOT_RUN_EXIT
    reservation: tuple[int, int] | None = None
    if args.profiler == "py-spy":
        try:
            reservation = _reserve_output(invocation.artifact)
        except FileExistsError:
            print(
                "profiling not run: output artifact already exists: "
                f"{invocation.artifact}",
                file=sys.stderr,
            )
            return NOT_RUN_EXIT
        except OSError as error:
            print(
                "profiling not run: could not reserve output artifact: "
                f"{type(error).__name__}",
                file=sys.stderr,
            )
            return NOT_RUN_EXIT
    print(
        f"diagnostic-only profiling artifact: {invocation.artifact}",
        file=sys.stderr,
    )
    try:
        result = _run_invocation(
            invocation,
            profiler=args.profiler,
            duration=args.duration,
            delay=args.delay,
        )
    except OSError as error:
        if reservation is not None:
            _remove_empty_reservation(invocation.artifact, reservation)
        print(
            f"profiling not run: could not launch {args.profiler}: "
            f"{type(error).__name__}",
            file=sys.stderr,
        )
        return NOT_RUN_EXIT
    if reservation is not None and (
        result != 0 or not _complete_outputs(invocation.reserved_outputs)
    ):
        _remove_empty_reservation(invocation.artifact, reservation)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
