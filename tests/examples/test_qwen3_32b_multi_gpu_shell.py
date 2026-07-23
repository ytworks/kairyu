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
