"""LiveCodeBench(+Pro)/SciCode: grading semantics and end-to-end degradation."""

import base64
import json
import pickle
import zlib

import httpx
from conftest import make_config, make_target

from evals.adapters import livecodebench as livecodebench_module
from evals.adapters.base import RunContext
from evals.adapters.livecodebench import (
    LiveCodeBenchAdapter,
    decode_private_tests,
    grade_code,
)
from evals.adapters.scicode import SciCodeAdapter
from evals.cache import BenchCache
from evals.runner import SuiteRunner
from evals.sandbox import ExecResult
from evals.store import ResultStore
from evals.types import BenchItem

STDIN_TESTS = [
    {"input": "2 3\n", "output": "5\n", "testtype": "stdin"},
    {"input": "-1 7\n", "output": "6\n", "testtype": "stdin"},
]
FUNCTIONAL_TESTS = [
    {"input": "4\n5", "output": "9", "testtype": "functional"},
    {"input": "-2\n2", "output": "0", "testtype": "functional"},
]


class _RecordingExecutionRunner:
    def __init__(
        self,
        *,
        modules: tuple[str, ...] = ("numpy",),
        available: tuple[bool, str] = (True, ""),
        stdout: str = "5\n",
    ) -> None:
        self.modules = set(modules)
        self.available = available
        self.stdout = stdout
        self.calls: list[dict] = []
        self.module_probes: list[str] = []

    def run_python(
        self,
        code: str,
        *,
        stdin: str = "",
        timeout_s: float = 30.0,
        memory_mb: int = 4096,
        files: dict[str, bytes] | None = None,
    ) -> ExecResult:
        self.calls.append(
            {
                "code": code,
                "stdin": stdin,
                "timeout_s": timeout_s,
                "memory_mb": memory_mb,
                "files": dict(files or {}),
            }
        )
        return ExecResult(
            returncode=0,
            stdout=self.stdout,
            stderr="",
            timed_out=False,
        )

    def has_module(self, name: str) -> bool:
        self.module_probes.append(name)
        return name in self.modules

    def metadata(self) -> dict:
        return {
            "identity": "recording-test-runner",
            "limits": {"network": "none", "filesystem": "ephemeral"},
        }

    def availability(self) -> tuple[bool, str]:
        return self.available


def test_grade_code_stdin_pass_and_fail():
    good = "a, b = map(int, input().split())\nprint(a + b)"
    passed, detail = grade_code(good, STDIN_TESTS, None)
    assert passed and detail == ""

    wrong = "a, b = map(int, input().split())\nprint(a - b)"
    passed, detail = grade_code(wrong, STDIN_TESTS, None)
    assert not passed
    assert "wrong answer" in detail


def test_grade_code_functional_pass_and_fail():
    good = "class Solution:\n    def add_numbers(self, a, b):\n        return a + b"
    passed, _ = grade_code(good, FUNCTIONAL_TESTS, "add_numbers")
    assert passed

    wrong = "class Solution:\n    def add_numbers(self, a, b):\n        return a * b"
    passed, detail = grade_code(wrong, FUNCTIONAL_TESTS, "add_numbers")
    assert not passed and "functional test" in detail


def test_grade_code_crash_and_timeout_reported(monkeypatch):
    passed, detail = grade_code("raise RuntimeError('x')", STDIN_TESTS[:1], None)
    assert not passed and "RuntimeError" in detail

    # The production timeout value is asserted through the injected-runner
    # contract below.  This test only needs to exercise the real timeout and
    # process-reap branch, not spend the complete service budget doing so.
    monkeypatch.setattr(livecodebench_module, "_TEST_TIMEOUT_S", 1.0)
    passed, detail = grade_code("import time; time.sleep(30)", STDIN_TESTS[:1], None)
    assert not passed and "timeout" in detail


def test_grade_code_uses_the_injected_execution_runner():
    execution_runner = _RecordingExecutionRunner()

    passed, detail = grade_code(
        "print('the fake runner owns execution')",
        STDIN_TESTS[:1],
        None,
        runner=execution_runner,
    )

    assert passed and detail == ""
    assert len(execution_runner.calls) == 1
    assert "fake runner owns execution" in execution_runner.calls[0]["code"]
    assert execution_runner.calls[0]["timeout_s"] == 6.0
    assert execution_runner.calls[0]["memory_mb"] == 4096


def test_decode_private_tests_both_encodings():
    tests = [{"input": "1", "output": "2", "testtype": "stdin"}]
    assert decode_private_tests(json.dumps(tests)) == tests
    blob = base64.b64encode(
        zlib.compress(pickle.dumps(json.dumps(tests)))
    ).decode()
    assert decode_private_tests(blob) == tests


def test_decode_private_tests_blocks_arbitrary_code(tmp_path):
    # M7: a hostile blob that pickles a global (arbitrary-code vector) must be
    # rejected, not unpickled at download time.
    import os

    import pytest

    evil = base64.b64encode(zlib.compress(pickle.dumps(os.system))).decode()
    with pytest.raises(pickle.UnpicklingError, match="blocked global"):
        decode_private_tests(evil)


def test_compose_solution_hoists_future_imports():
    # M8: a model solution starting with `from __future__ import ...` must keep
    # it as the first statement even after the import header is prepended.
    from evals.adapters.livecodebench import _compose_solution

    composed = _compose_solution("from __future__ import annotations\nx = 1\n")
    assert composed.startswith("from __future__ import annotations\n")
    # it must be a valid module (no "future import must be first" SyntaxError)
    compile(composed, "<solution>", "exec")


async def test_livecodebench_end_to_end_with_correct_model(tmp_path):
    """A canned 'model' that answers the fixture problems correctly scores 1.0."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        prompt = body["messages"][0]["content"]
        if "their sum" in prompt:
            code = "a, b = map(int, input().split())\nprint(a + b)"
        else:
            code = "class Solution:\n    def add_numbers(self, a, b):\n        return a + b"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"Here you go:\n```python\n{code}\n```",
                        },
                    }
                ]
            },
        )

    config = make_config(tmp_path, models=("m",), only=("livecodebench",))
    runner = SuiteRunner(
        config,
        http_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        probe_docker=lambda: (False, "t"),
    )
    assert await runner.run() == 0
    pair = ResultStore(tmp_path / "results", "test-run").load_pair("livecodebench", "m")
    assert pair.status == "completed"
    assert pair.score == 1.0
    assert isinstance(pair.methodology["execution"]["runner"], dict)
    assert pair.methodology["execution"]["per_test_limits"] == {
        "timeout_s": 6.0,
        "memory_mb": 4096,
    }


async def test_livecodebench_mock_gateway_scores_zero_not_crash(tmp_path, http_factory):
    config = make_config(tmp_path, models=("m",), only=("livecodebench", "livecodebench-pro"))
    runner = SuiteRunner(config, http_factory=http_factory, probe_docker=lambda: (False, "t"))
    assert await runner.run() == 0
    for name in ("livecodebench", "livecodebench-pro"):
        pair = ResultStore(tmp_path / "results", "test-run").load_pair(name, "m")
        assert pair.status == "completed"
        assert pair.score == 0.0  # mock text has no code block


async def test_livecodebench_empty_completions_are_failed_not_scored_zero(tmp_path):
    """No generated answer is missing evidence, not an incorrect solution."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": ""},
                        "finish_reason": "length",
                    }
                ]
            },
        )

    config = make_config(tmp_path, models=("m",), only=("livecodebench",))
    runner = SuiteRunner(
        config,
        http_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        probe_docker=lambda: (False, "t"),
    )
    assert await runner.run() == 1

    pair = ResultStore(tmp_path / "results", "test-run").load_pair("livecodebench", "m")
    assert pair.status == "failed"
    assert pair.score is None
    assert pair.metrics["n_scored"] == 0
    assert pair.metrics["n_failed"] == pair.metrics["n_total"]
    assert pair.items
    assert all(item.status == "failed" for item in pair.items)
    assert all("empty completion" in (item.error or "") for item in pair.items)
    assert all(item.latency_s is not None for item in pair.items)


def _scicode_ctx(tmp_path, execution_runner=None) -> RunContext:
    kwargs = dict(
        cache=BenchCache(tmp_path / "cache"),
        http_factory=lambda: httpx.AsyncClient(),
        offline_fixtures=True,
    )
    if execution_runner is not None:
        kwargs["execution_runner"] = execution_runner
    return RunContext(**kwargs)


async def test_scicode_scores_correct_step(tmp_path):
    adapter = SciCodeAdapter()
    ctx = _scicode_ctx(tmp_path)
    item = BenchItem(
        id="scicode-fx.1",
        payload={
            "step_id": "fx.1",
            "dependencies": "import numpy as np",
            "prior_code": "",
            "test_cases": [
                "assert abs(vector_norm(np.array([3.0, 4.0])) - 5.0) < 1e-9"
            ],
        },
    )
    good = (
        "```python\ndef vector_norm(v):\n"
        "    return float(np.sqrt(np.sum(np.asarray(v) ** 2)))\n```"
    )
    result = await adapter.score(item, good, ctx)
    assert result.status == "completed" and result.score == 1.0

    result = await adapter.score(item, "```python\ndef vector_norm(v):\n    return 0\n```", ctx)
    assert result.score == 0.0


async def test_adapters_use_and_disclose_the_selected_execution_runner(tmp_path):
    execution_runner = _RecordingExecutionRunner()
    ctx = _scicode_ctx(tmp_path, execution_runner)
    lcb_item = BenchItem(
        id="lcb-fx",
        payload={
            "tests": STDIN_TESTS[:1],
            "fn_name": None,
        },
    )
    lcb_result = await LiveCodeBenchAdapter().score(
        lcb_item,
        "```python\nprint('selected runner')\n```",
        ctx,
    )
    scicode_item = BenchItem(
        id="scicode-fx.1",
        payload={
            "step_id": "fx.1",
            "dependencies": "",
            "prior_code": "",
            "test_cases": ["assert selected_runner_was_used()"],
        },
    )

    scicode_result = await SciCodeAdapter().score(
        scicode_item,
        "```python\ndef selected_runner_was_used():\n    return True\n```",
        ctx,
    )

    assert lcb_result.status == "completed" and lcb_result.score == 1.0
    assert scicode_result.status == "completed" and scicode_result.score == 1.0
    assert len(execution_runner.calls) == 2
    assert "selected runner" in execution_runner.calls[0]["code"]
    assert "selected_runner_was_used" in execution_runner.calls[1]["code"]
    assert SciCodeAdapter().methodology(ctx)["execution"]["runner"] == (
        execution_runner.metadata()
    )
    assert LiveCodeBenchAdapter().methodology(ctx)["execution"]["runner"] == (
        execution_runner.metadata()
    )


def test_scicode_probes_the_selected_runners_modules(tmp_path):
    execution_runner = _RecordingExecutionRunner(modules=())
    ctx = _scicode_ctx(tmp_path, execution_runner)

    reason = SciCodeAdapter().check_preconditions(make_target(), ctx)

    assert execution_runner.module_probes == ["numpy"]
    assert reason == "selected execution runner lacks numpy (pip install numpy)"


def test_execution_adapters_fail_closed_when_the_selected_runner_is_unavailable(
    tmp_path,
):
    execution_runner = _RecordingExecutionRunner(
        available=(False, "container runtime is down")
    )
    ctx = _scicode_ctx(tmp_path, execution_runner)

    lcb_reason = LiveCodeBenchAdapter().check_preconditions(make_target(), ctx)
    scicode_reason = SciCodeAdapter().check_preconditions(make_target(), ctx)

    assert lcb_reason == (
        "selected execution runner unavailable (container runtime is down)"
    )
    assert scicode_reason == lcb_reason
    assert execution_runner.module_probes == []


async def test_scicode_target_tests_without_golden_data_are_unjudged(tmp_path):
    adapter = SciCodeAdapter()
    ctx = _scicode_ctx(tmp_path)
    item = BenchItem(
        id="scicode-fx.9",
        payload={
            "step_id": "fx.9",
            "dependencies": "",
            "prior_code": "",
            "test_cases": ["assert f(1) == target"],
        },
    )
    result = await adapter.score(item, "```python\ndef f(x):\n    return x\n```", ctx)
    assert result.status == "unjudged"
    assert "test_data.h5" in result.error


async def test_scicode_end_to_end_on_fixtures(tmp_path, http_factory):
    config = make_config(tmp_path, models=("m",), only=("scicode",))
    runner = SuiteRunner(config, http_factory=http_factory, probe_docker=lambda: (False, "t"))
    assert await runner.run() == 0
    pair = ResultStore(tmp_path / "results", "test-run").load_pair("scicode", "m")
    assert pair.status == "completed"  # numpy present in this venv
    assert pair.score == 0.0  # mock emits no code
