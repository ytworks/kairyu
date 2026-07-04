"""LiveCodeBench(+Pro)/SciCode: grading semantics and end-to-end degradation."""

import base64
import json
import pickle
import zlib

import httpx
from conftest import make_config

from kairyu.bench.adapters.base import RunContext
from kairyu.bench.adapters.livecodebench import decode_private_tests, grade_code
from kairyu.bench.adapters.scicode import SciCodeAdapter
from kairyu.bench.cache import BenchCache
from kairyu.bench.runner import SuiteRunner
from kairyu.bench.store import ResultStore
from kairyu.bench.types import BenchItem

STDIN_TESTS = [
    {"input": "2 3\n", "output": "5\n", "testtype": "stdin"},
    {"input": "-1 7\n", "output": "6\n", "testtype": "stdin"},
]
FUNCTIONAL_TESTS = [
    {"input": "4\n5", "output": "9", "testtype": "functional"},
    {"input": "-2\n2", "output": "0", "testtype": "functional"},
]


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


def test_grade_code_crash_and_timeout_reported():
    passed, detail = grade_code("raise RuntimeError('x')", STDIN_TESTS[:1], None)
    assert not passed and "RuntimeError" in detail

    passed, detail = grade_code("import time; time.sleep(30)", STDIN_TESTS[:1], None)
    assert not passed and "timeout" in detail


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
    from kairyu.bench.adapters.livecodebench import _compose_solution

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
    assert "sandbox" in pair.methodology["execution"]


async def test_livecodebench_mock_gateway_scores_zero_not_crash(tmp_path, http_factory):
    config = make_config(tmp_path, models=("m",), only=("livecodebench", "livecodebench-pro"))
    runner = SuiteRunner(config, http_factory=http_factory, probe_docker=lambda: (False, "t"))
    assert await runner.run() == 0
    for name in ("livecodebench", "livecodebench-pro"):
        pair = ResultStore(tmp_path / "results", "test-run").load_pair(name, "m")
        assert pair.status == "completed"
        assert pair.score == 0.0  # mock text has no code block


def _scicode_ctx(tmp_path) -> RunContext:
    return RunContext(
        cache=BenchCache(tmp_path / "cache"),
        http_factory=lambda: httpx.AsyncClient(),
        offline_fixtures=True,
    )


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
