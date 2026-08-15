"""Sandbox runner report/validation logic (no Docker, no subprocess).

Silent mis-parsing here would convert failing tests into "passed" evidence fed
to the synthesis/verifier stages, so the summarization contract is pinned.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "examples/qwen3.8-deepseek-v4-8gpu/sandbox/runner.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("sandbox_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _submission(**overrides):
    body = {
        "schema_version": 1,
        "language": "python",
        "files": {
            "solution.py": "x = 1",
            "test_solution.py": "def test_x():\n    assert True",
        },
        "entry": {"kind": "pytest"},
    }
    body.update(overrides)
    return body


def test_parse_rejects_non_python_language_as_422():
    with pytest.raises(runner.SubmissionError) as excinfo:
        runner.parse_submission(_submission(language="javascript"))
    assert excinfo.value.status_code == 422


def test_parse_rejects_path_escapes_and_requires_one_test_file():
    with pytest.raises(runner.SubmissionError):
        runner.parse_submission(
            _submission(files={"../evil.py": "x", "test_a.py": "y"})
        )
    with pytest.raises(runner.SubmissionError):
        runner.parse_submission(_submission(files={"solution.py": "x"}))
    with pytest.raises(runner.SubmissionError):
        runner.parse_submission(
            _submission(
                files={"test_a.py": "x", "test_b.py": "y", "solution.py": "z"}
            )
        )
    parsed = runner.parse_submission(_submission())
    assert parsed["test_file"] == "test_solution.py"
    assert parsed["limits"]["wall_time_s"] == 10.0


def test_summarize_folds_pytest_report_counts():
    report = runner.summarize(
        {
            "exit_code": 1,
            "tests": [
                {"id": "test_solution.py::test_a", "outcome": "passed"},
                {"id": "test_solution.py::test_b", "outcome": "failed"},
                {"id": "test_solution.py::test_c", "outcome": "error"},
                {"id": 5, "outcome": "passed"},  # malformed row is dropped
            ],
        },
        timed_out=False,
        returncode=0,
        stdout=b"out",
        stderr=b"",
        output_bytes=1024,
        duration_ms=10,
    )
    assert report["status"] == "ok"
    assert (report["passed"], report["failed"], report["errors"]) == (1, 1, 1)
    assert len(report["tests"]) == 3


def test_summarize_missing_report_is_setup_error_never_ok():
    report = runner.summarize(
        None,
        timed_out=False,
        returncode=2,
        stdout=b"",
        stderr=b"import error",
        output_bytes=1024,
        duration_ms=5,
    )
    assert report["status"] == "setup_error"
    assert report["passed"] == 0


@pytest.mark.parametrize("exit_code", [2, 3, 4, 5])
def test_summarize_pytest_abnormal_exit_is_setup_error(exit_code):
    report = runner.summarize(
        {
            "exit_code": exit_code,
            "tests": [{"id": "test_solution.py::test_a", "outcome": "passed"}],
        },
        timed_out=False,
        returncode=0,
        stdout=b"",
        stderr=b"",
        output_bytes=1024,
        duration_ms=5,
    )

    assert report["status"] == "setup_error"
    assert report["detail"] == f"returncode=0; pytest_exit_code={exit_code}"


def test_summarize_shim_failure_with_report_is_setup_error():
    report = runner.summarize(
        {
            "exit_code": 0,
            "tests": [{"id": "test_solution.py::test_a", "outcome": "passed"}],
        },
        timed_out=False,
        returncode=-9,
        stdout=b"",
        stderr=b"",
        output_bytes=1024,
        duration_ms=5,
    )

    assert report["status"] == "setup_error"
    assert report["detail"] == "returncode=-9; pytest_exit_code=0"


def test_summarize_empty_test_report_is_setup_error():
    report = runner.summarize(
        {"exit_code": 0, "tests": []},
        timed_out=False,
        returncode=0,
        stdout=b"",
        stderr=b"",
        output_bytes=1024,
        duration_ms=5,
    )

    assert report["status"] == "setup_error"
    assert report["detail"] == "no_tests_collected"


def test_summarize_timeout_and_output_truncation():
    report = runner.summarize(
        {"exit_code": 0, "tests": []},
        timed_out=True,
        returncode=-9,
        stdout=b"a" * 100,
        stderr=b"",
        output_bytes=16,
        duration_ms=999,
    )
    assert report["status"] == "timeout"
    assert report["resource"]["timed_out"] is True
    assert report["resource"]["output_truncated"] is True
    assert report["stdout_excerpt"].endswith("[truncated]")
