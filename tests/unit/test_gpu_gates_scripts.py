"""m19 D3: every gpu_gates script dry-runs and references REAL files/tests."""

import re
import subprocess
from pathlib import Path

import pytest

SCRIPTS = sorted(Path("scripts/gpu_gates").glob("[0-9g]*.sh"))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_dry_run_emits_commands(script):
    result = subprocess.run(
        ["bash", str(script), "--dry-run"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    commands = [line for line in result.stdout.splitlines() if line.startswith("+ ")]
    assert commands, f"{script.name} emitted no commands"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_referenced_paths_exist(script):
    """Deploy day must not discover missing files: every tests/, scripts/,
    bench/ and deploy/ path a script mentions exists TODAY."""
    text = script.read_text()
    for match in re.findall(r"(?:tests|scripts|bench|deploy)/[\w./-]+", text):
        path = Path(match)
        assert path.exists(), f"{script.name} references missing {match}"


def test_gpu_marker_tests_exist_and_are_deselected():
    gpu_tests = list(Path("tests/gpu").glob("test_*.py"))
    assert gpu_tests, "tests/gpu mirror is empty"
    result = subprocess.run(
        ["uv", "run", "pytest", "tests/gpu", "--collect-only", "-q", "--no-cov"],
        capture_output=True, text=True,
    )
    assert "deselected" in result.stdout  # addopts excludes gpu by default
