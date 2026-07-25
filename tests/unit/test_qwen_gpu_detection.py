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


def _gpu_list_stub(count: int) -> str:
    lines = "\n".join(str(index) for index in range(count))
    return f"#!/bin/sh\ncat <<'EOF'\n{lines}\nEOF\n"


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
