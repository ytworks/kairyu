"""Regression tests for explicit, truthful pytest suite selection."""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_portable_collection_includes_distributed_tests() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "--no-cov",
            "tests",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == pytest.ExitCode.OK, result.stderr
    assert "tests/dist/test_distributed.py" in result.stdout
    assert "tests/dist/test_pd_two_process.py" in result.stdout


def test_portable_collection_deselects_declared_environment_suites() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "--no-cov",
            "tests/unit/test_postgres_batch_store.py",
            "tests/unit/test_fleet_elastic.py",
            "tests/compat/test_vllm_backend.py",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == pytest.ExitCode.OK, result.stderr
    assert "test_helm_chart_renders" not in result.stdout
    assert "test_generate_roundtrip_with_real_vllm" not in result.stdout
    assert "test_postgres_batch_store.py::" not in result.stdout
