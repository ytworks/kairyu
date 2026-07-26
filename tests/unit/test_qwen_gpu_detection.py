"""GPU-count detection must not mistake a broken driver for "no GPUs".

`nvidia-smi ... | wc -l` takes the PIPELINE's exit status, which is `wc`'s
success. When `nvidia-smi` itself fails — exit 18 is what a driver upgrade
without a kernel-module reload produces — the count reads as 0 and the operator
is told the host has the wrong NUMBER of GPUs. It does not; the driver is
unusable. These run the real scripts against a stubbed `nvidia-smi`.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "qwen3-32b-multi-gpu"

# verbatim from the host that hit this: nvidia-smi exits 18 and prints to stderr
DRIVER_MISMATCH = """\
#!/bin/sh
echo 'Failed to initialize NVML: Driver/library version mismatch' >&2
echo 'NVML library version: 595.84' >&2
exit 18
"""


def _stub_bin(tmp_path: Path, name: str, body: str) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / name
    stub.write_text(body)
    stub.chmod(0o755)
    return bin_dir


def _run(script: Path, bin_dir: Path, cwd: Path | None = None):
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}")
    return subprocess.run(
        ["sh", str(script)],
        env=env,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _gpu_list_stub(count: int, *, warning: str | None = None) -> str:
    lines = "\n".join(str(index) for index in range(count))
    body = "#!/bin/sh\n"
    if warning is not None:
        # nvidia-smi warns on stderr; merging it into the captured list is how a
        # date or bus id becomes an extra "GPU"
        body += f"echo {warning!r} >&2\n"
    body += f"cat <<'EOF'\n{lines}\nEOF\n" if count else "true\n"
    return body


# verbatim shape of a real nvidia-smi warning: every field here contains digits
NOISY_WARNING = (
    "WARNING: infoROM is corrupted at gpu 0000:16:00.0 (driver 595.84, 2026-07-25)"
)


@pytest.fixture
def docker_stub_dir(tmp_path):
    """`docker` that records its arguments instead of starting anything."""
    bin_dir = _stub_bin(tmp_path, "docker", "#!/bin/sh\necho \"docker $*\" >> \"$RECORD\"\n")
    return bin_dir


def test_failed_nvidia_smi_is_reported_as_a_driver_fault(tmp_path, docker_stub_dir):
    _stub_bin(tmp_path, "nvidia-smi", DRIVER_MISMATCH)
    result = _run(EXAMPLE / "run.sh", docker_stub_dir)

    assert result.returncode != 0
    # the actual driver error must reach the operator...
    assert "driver is not usable" in result.stderr
    assert "Driver/library version mismatch" in result.stderr
    # ...and it must NOT be reported as a GPU-count problem
    assert "found 0" not in result.stderr


def test_a_working_driver_with_eight_gpus_starts_compose(tmp_path, docker_stub_dir):
    _stub_bin(tmp_path, "nvidia-smi", _gpu_list_stub(8))
    record = tmp_path / "docker-calls"
    env_before = os.environ.get("RECORD")
    os.environ["RECORD"] = str(record)
    try:
        result = _run(EXAMPLE / "run.sh", docker_stub_dir)
    finally:
        if env_before is None:
            os.environ.pop("RECORD", None)
        else:  # pragma: no cover - only when the caller had one
            os.environ["RECORD"] = env_before

    assert result.returncode == 0, result.stderr
    assert "Using all 8 visible GPUs" in result.stdout
    assert "compose up" in record.read_text()


def test_an_unsupported_gpu_count_still_says_so(tmp_path, docker_stub_dir):
    # the pre-existing guard must survive: 3 GPUs is a real count problem
    _stub_bin(tmp_path, "nvidia-smi", _gpu_list_stub(3))
    result = _run(EXAMPLE / "run.sh", docker_stub_dir)

    assert result.returncode != 0
    assert "found 3" in result.stderr
    assert "driver is not usable" not in result.stderr


def test_container_startup_command_also_fails_loudly(tmp_path):
    """The same guard runs inside the container, where a stale CDI spec or a
    missing device produces the identical nvidia-smi failure."""
    yaml = pytest.importorskip("yaml")

    spec = yaml.safe_load((EXAMPLE / "compose.yaml").read_text())
    # compose escapes shell `$` as `$$`; undo it to run the real command
    command = spec["services"]["kairyu"]["command"][0].replace("$$", "$")
    script = tmp_path / "startup.sh"
    script.write_text(command)

    bin_dir = _stub_bin(tmp_path, "nvidia-smi", DRIVER_MISMATCH)
    result = subprocess.run(
        ["sh", "-ec", command],
        env=dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}"),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode != 0
    assert "GPUs are not usable here" in result.stderr
    assert "Driver/library version mismatch" in result.stderr
    assert "found 0" not in result.stderr


@pytest.mark.parametrize(
    "script", ["run.sh", "benchmark.sh"], ids=["run", "benchmark"]
)
def test_no_script_counts_gpus_through_a_swallowing_pipeline(script):
    text = (EXAMPLE / script).read_text(encoding="utf-8")
    assert "--format=csv,noheader | wc -l" not in text, (
        f"{script} pipes nvidia-smi into wc -l again; the pipeline's status is "
        "wc's, so a driver failure becomes a GPU count of 0"
    )


def test_compose_startup_does_not_count_through_a_swallowing_pipeline():
    text = (EXAMPLE / "compose.yaml").read_text(encoding="utf-8")
    assert "--format=csv,noheader | wc -l" not in text


def test_scripts_are_executable():
    for name in ("run.sh", "benchmark.sh"):
        assert shutil.which(str(EXAMPLE / name)) or os.access(EXAMPLE / name, os.X_OK)


def test_a_digit_bearing_warning_is_not_counted_as_a_gpu(tmp_path, docker_stub_dir):
    # review [P1] on #127: `grep -c '[0-9]'` counted any line containing a digit,
    # so a warning line plus indices 0 and 1 reported THREE GPUs and the script
    # refused to start a perfectly good 2-GPU host
    _stub_bin(tmp_path, "nvidia-smi", _gpu_list_stub(2, warning=NOISY_WARNING))
    record = tmp_path / "docker-calls"
    os.environ["RECORD"] = str(record)
    try:
        result = _run(EXAMPLE / "run.sh", docker_stub_dir)
    finally:
        os.environ.pop("RECORD", None)

    assert result.returncode == 0, result.stderr
    assert "Using all 2 visible GPUs" in result.stdout
    assert "found 3" not in result.stderr


def test_a_warning_on_stdout_fails_loudly_rather_than_miscounting(
    tmp_path, docker_stub_dir
):
    _stub_bin(
        tmp_path,
        "nvidia-smi",
        f"#!/bin/sh\necho '{NOISY_WARNING}'\nprintf '0\\n1\\n'\n",
    )
    result = _run(EXAMPLE / "run.sh", docker_stub_dir)

    assert result.returncode != 0
    assert "not GPU indices" in result.stderr
    assert "found 3" not in result.stderr


def test_empty_output_reports_zero_instead_of_dying_on_set_e(tmp_path, docker_stub_dir):
    # `grep -c` exits 1 with no matches, and `set -e` killed the script before the
    # found-0 diagnostic could run
    _stub_bin(tmp_path, "nvidia-smi", _gpu_list_stub(0))
    result = _run(EXAMPLE / "run.sh", docker_stub_dir)

    assert result.returncode != 0
    assert "found 0" in result.stderr, result.stderr


def test_container_startup_command_counts_the_same_way(tmp_path):
    """The compose command must not diverge from run.sh — it is the one that
    actually decides tensor_parallel_size."""
    yaml = pytest.importorskip("yaml")

    spec = yaml.safe_load((EXAMPLE / "compose.yaml").read_text())
    command = spec["services"]["kairyu"]["command"][0].replace("$$", "$")
    bin_dir = _stub_bin(tmp_path, "nvidia-smi", _gpu_list_stub(2, warning=NOISY_WARNING))
    result = subprocess.run(
        ["sh", "-ec", command],
        env=dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}"),
        capture_output=True,
        text=True,
        timeout=60,
    )
    # it gets PAST the count guard (and then fails on the container-only config
    # template, which is as far as this can run outside the image)
    assert "found 3" not in result.stderr, result.stderr
    assert "not GPU indices" not in result.stderr
    assert "config.template.yaml" in result.stderr


def test_no_site_counts_with_a_digit_anywhere_match():
    for name in ("run.sh", "benchmark.sh"):
        text = (EXAMPLE / name).read_text(encoding="utf-8")
        assert "grep -c '[0-9]'" not in text, name
    assert "grep -c '[0-9]'" not in (EXAMPLE / "compose.yaml").read_text(encoding="utf-8")


def _benchmark_stubs(tmp_path: Path, gpu_output: str, *, gpu_ok: bool = True) -> Path:
    """`docker` and `curl` stubs so benchmark.sh reaches its GPU-count guard.

    `docker compose exec ... nvidia-smi` is the only docker call before the
    guard; curl serves the readyz probe.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    status = "0" if gpu_ok else "1"
    (bin_dir / "docker").write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        "  *nvidia-smi*)\n"
        f"    cat <<'EOF'\n{gpu_output}\nEOF\n"
        f"    exit {status}\n"
        "    ;;\n"
        "  *images*) echo sha256:stub ;;\n"
        "  *) : ;;\n"
        "esac\n"
    )
    (bin_dir / "curl").write_text("#!/bin/sh\nexit 0\n")
    for name in ("docker", "curl"):
        (bin_dir / name).chmod(0o755)
    return bin_dir


@pytest.mark.parametrize(
    ("gpu_output", "expected"),
    [("", "found 0"), ("0\n1\n2", "found 3")],
    ids=["empty", "unsupported-count"],
)
def test_benchmark_refuses_to_record_a_bad_gpu_count(tmp_path, gpu_output, expected):
    # review [P1] on #127: benchmark.sh parsed an empty list cleanly to 0 and then
    # printed "GPUs/TP=0" in the header AND passed --tensor-parallel 0
    bin_dir = _benchmark_stubs(tmp_path, gpu_output)
    result = _run(EXAMPLE / "benchmark.sh", bin_dir)

    assert result.returncode != 0
    assert expected in result.stderr, result.stderr
    assert "GPUs/TP=" not in result.stdout


def test_benchmark_rejects_a_digit_bearing_line_from_the_container(tmp_path):
    bin_dir = _benchmark_stubs(tmp_path, f"{NOISY_WARNING}\n0\n1")
    result = _run(EXAMPLE / "benchmark.sh", bin_dir)

    assert result.returncode != 0
    assert "not GPU indices" in result.stderr
    assert "found 3" not in result.stderr


def test_benchmark_accepts_a_supported_count(tmp_path):
    bin_dir = _benchmark_stubs(tmp_path, "0\n1")
    result = _run(EXAMPLE / "benchmark.sh", bin_dir)

    # it gets past the guard (and then fails later on the stubbed environment)
    assert "found 0" not in result.stderr
    assert "requires 2, 4, or 8" not in result.stderr
