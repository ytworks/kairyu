"""Execution sandbox: success, failure, timeout, memory, file placement."""

import sys

import pytest

from kairyu.bench.sandbox import has_module, run_python


def test_success_captures_stdout():
    result = run_python("print(21 * 2)")
    assert result.ok
    assert result.stdout.strip() == "42"


def test_stdin_is_delivered():
    result = run_python("import sys; print(sys.stdin.read().strip().upper())", stdin="hi")
    assert result.stdout.strip() == "HI"


def test_exception_sets_returncode_and_stderr():
    result = run_python("raise ValueError('boom')")
    assert not result.ok
    assert result.returncode != 0
    assert "boom" in result.stderr


def test_wall_clock_timeout():
    result = run_python("import time; time.sleep(30)", timeout_s=1.0)
    assert result.timed_out
    assert not result.ok


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="macOS rejects RLIMIT_AS/RLIMIT_DATA with EINVAL; containment is Linux-only",
)
def test_memory_bomb_is_contained():
    result = run_python("x = bytearray(10**10)", timeout_s=20.0, memory_mb=256)
    assert not result.ok  # MemoryError or killed, never host OOM


def test_files_are_placed_in_cwd():
    result = run_python(
        "print(open('data.txt').read())", files={"data.txt": b"payload"}
    )
    assert result.stdout.strip() == "payload"


def test_env_is_scrubbed():
    result = run_python("import os; print(sorted(os.environ))")
    assert result.ok
    assert "HF_TOKEN" not in result.stdout


def test_has_module():
    assert has_module("json")
    assert not has_module("definitely_not_a_module_xyz")
