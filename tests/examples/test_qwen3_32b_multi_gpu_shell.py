from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "qwen3-32b-multi-gpu"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _run(
    script: Path, fake_bin: Path, state_dir: Path, **extra: str
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "STATE_DIR": str(state_dir),
        "PROGRESS_INTERVAL_S": "5",
        **extra,
    }
    return subprocess.run(
        ["/bin/sh", str(script)],
        cwd=script.parent,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )


def _readiness_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    example = tmp_path / "example"
    fake_bin = tmp_path / "bin"
    state = tmp_path / "state"
    example.mkdir()
    fake_bin.mkdir()
    state.mkdir()
    shutil.copy(EXAMPLE / "run-benchmark.sh", example / "run-benchmark.sh")
    _write_executable(example / "run.sh", "#!/bin/sh\necho run-called\n")
    _write_executable(example / "benchmark.sh", "#!/bin/sh\necho benchmark-called\n")
    _write_executable(fake_bin / "sleep", "#!/bin/sh\nexit 0\n")
    return example, fake_bin, state


def test_run_benchmark_reports_readiness_wait_progress(tmp_path: Path) -> None:
    example, fake_bin, state = _readiness_fixture(tmp_path)
    _write_executable(
        fake_bin / "curl",
        """#!/bin/sh
count_file="$STATE_DIR/curl-count"
count=0
[ ! -f "$count_file" ] || count="$(cat "$count_file")"
count=$((count + 1))
printf '%s\n' "$count" >"$count_file"
[ "$count" -ge 3 ]
""",
    )

    result = _run(example / "run-benchmark.sh", fake_bin, state)

    assert result.returncode == 0, result.stderr
    assert "[startup] starting Qwen3-32B service" in result.stdout
    assert "[startup] waiting for readiness" in result.stdout
    assert "[startup] waiting for readiness (elapsed 5s)" in result.stdout
    assert "[startup] waiting for readiness (elapsed 10s)" in result.stdout
    assert "[startup] ready after 10s" in result.stdout
    assert "[benchmark] starting" in result.stdout
    assert "benchmark-called" in result.stdout


def test_run_benchmark_timeout_keeps_compose_diagnostics(tmp_path: Path) -> None:
    example, fake_bin, state = _readiness_fixture(tmp_path)
    _write_executable(fake_bin / "curl", "#!/bin/sh\nexit 1\n")
    _write_executable(
        fake_bin / "docker",
        """#!/bin/sh
[ "${1:-}" = compose ] && [ "${2:-}" = logs ] && echo compose-diagnostics >&2
""",
    )

    result = _run(example / "run-benchmark.sh", fake_bin, state)

    assert result.returncode == 1
    assert "[startup] waiting for readiness (elapsed 900s)" in result.stdout
    assert "Kairyu did not become ready" in result.stderr
    assert "compose-diagnostics" in result.stderr
    assert "benchmark-called" not in result.stdout


def _benchmark_fixture(
    tmp_path: Path, *, metrics: bool = True
) -> tuple[Path, Path, Path]:
    example = tmp_path / "example"
    fake_bin = tmp_path / "bin"
    state = tmp_path / "state"
    example.mkdir()
    fake_bin.mkdir()
    state.mkdir()
    (example / "results").mkdir()
    shutil.copy(EXAMPLE / "benchmark.sh", example / "benchmark.sh")
    (example / "benchmark_report.py").write_text("", encoding="utf-8")
    metrics_body = (
        """count_file="$STATE_DIR/metrics-count"
count=0
[ ! -f "$count_file" ] || count="$(cat "$count_file")"
count=$((count + 1))
printf '%s\n' "$count" >"$count_file"
if [ "$count" -eq 1 ]; then value=100; else value=150; fi
printf 'kairyu_requests_total{code="200",model="qwen3-32b"} %s.0\n' "$value"
exit 0
"""
        if metrics
        else "exit 1\n"
    )
    _write_executable(
        fake_bin / "curl",
        f"""#!/bin/sh
case "$*" in
  *readyz*) exit 0 ;;
  *metrics*) {metrics_body}  ;;
esac
exit 1
""",
    )
    _write_executable(
        fake_bin / "sleep",
        """#!/bin/sh
count_file="$STATE_DIR/sleep-count"
count=0
[ ! -f "$count_file" ] || count="$(cat "$count_file")"
count=$((count + 1))
printf '%s\n' "$count" >"$count_file"
[ "$count" -lt 2 ] || touch "$STATE_DIR/progress-observed"
/bin/sleep 0.02
""",
    )
    _write_executable(
        fake_bin / "docker",
        """#!/bin/sh
if [ "${1:-}" = compose ] && [ "${2:-}" = exec ]; then
  echo 4
  exit 0
fi
if [ "${1:-}" = compose ] && [ "${2:-}" = images ]; then
  echo sha256:test
  exit 0
fi
case "$*" in
  *benchmark_report.py*)
    touch "$STATE_DIR/report-called"
    echo report-output
    exit 0
    ;;
esac
while [ ! -f "$STATE_DIR/progress-observed" ]; do /bin/sleep 0.01; done
echo benchmark-output
exit "${BENCHMARK_EXIT:-0}"
""",
    )
    return example, fake_bin, state


def test_benchmark_reports_metric_progress_and_configuration(tmp_path: Path) -> None:
    example, fake_bin, state = _benchmark_fixture(tmp_path)

    result = _run(
        example / "benchmark.sh",
        fake_bin,
        state,
        NUM_REQUESTS="3",
        CONCURRENCY="2",
        MAX_TOKENS="16",
    )

    assert result.returncode == 0, result.stderr
    assert (
        "[benchmark] requests=3 concurrency=2 max_tokens=16 GPUs/TP=4"
        in result.stdout
    )
    assert "[benchmark] completed 3/3 (elapsed 5s)" in result.stdout
    assert "benchmark-output" in result.stdout
    assert "[report] generating Markdown report" in result.stdout
    assert "report-output" in result.stdout


def test_benchmark_falls_back_when_metrics_are_unavailable(tmp_path: Path) -> None:
    example, fake_bin, state = _benchmark_fixture(tmp_path, metrics=False)

    result = _run(example / "benchmark.sh", fake_bin, state, NUM_REQUESTS="3")

    assert result.returncode == 0, result.stderr
    assert "[benchmark] running (elapsed 5s)" in result.stdout


def test_benchmark_falls_back_when_metrics_are_malformed(tmp_path: Path) -> None:
    example, fake_bin, state = _benchmark_fixture(tmp_path, metrics=False)
    _write_executable(
        fake_bin / "curl",
        """#!/bin/sh
case "$*" in
  *readyz*) exit 0 ;;
  *metrics*)
    echo 'kairyu_requests_total{code="200",model="qwen3-32b"} not-a-number'
    exit 0
    ;;
esac
exit 1
""",
    )

    result = _run(example / "benchmark.sh", fake_bin, state, NUM_REQUESTS="3")

    assert result.returncode == 0, result.stderr
    assert "[benchmark] running (elapsed 5s)" in result.stdout


def test_benchmark_failure_stays_failure_and_skips_report(tmp_path: Path) -> None:
    example, fake_bin, state = _benchmark_fixture(tmp_path)

    result = _run(
        example / "benchmark.sh",
        fake_bin,
        state,
        NUM_REQUESTS="3",
        BENCHMARK_EXIT="7",
    )

    assert result.returncode == 7
    assert not (state / "report-called").exists()
    assert "[report] generating Markdown report" not in result.stdout
