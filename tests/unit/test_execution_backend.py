"""HttpExecutionBackend degrade contract and code-block extraction (ECO-D3)."""

import json

import httpx

from kairyu.orchestration.execution import (
    ExecutionLimits,
    ExecutionRequest,
    HttpExecutionBackend,
    extract_python_block,
)


def _request() -> ExecutionRequest:
    return ExecutionRequest(
        request_id="r1",
        language="python",
        files={"solution.py": "x = 1", "test_solution.py": "def test(): pass"},
        limits=ExecutionLimits(wall_time_s=2.0),
    )


def _backend(handler) -> HttpExecutionBackend:
    return HttpExecutionBackend(
        base_url="http://executor:8080",
        transport=httpx.MockTransport(handler),
        excerpt_bytes=64,
    )


async def test_connection_error_degrades_to_unavailable():
    def handler(request):
        raise httpx.ConnectError("refused")

    report = await _backend(handler).execute(_request())
    assert report.status == "unavailable"
    assert report.detail == "ConnectError"


async def test_server_error_degrades_to_unavailable():
    def handler(request):
        return httpx.Response(500, text="boom")

    report = await _backend(handler).execute(_request())
    assert report.status == "unavailable"
    assert report.detail == "http_500"


async def test_double_429_degrades_after_one_retry():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429)

    report = await _backend(handler).execute(_request())
    assert report.status == "unavailable"
    assert len(calls) == 2


async def test_unsupported_language_maps_to_unsupported():
    def handler(request):
        return httpx.Response(422, json={"error": "unsupported_language"})

    report = await _backend(handler).execute(_request())
    assert report.status == "unsupported"


async def test_ok_report_parses_tests_and_caps_excerpts():
    def handler(request):
        body = json.loads(request.content)
        assert body["schema_version"] == 1
        assert body["limits"]["wall_time_s"] == 2.0
        return httpx.Response(
            200,
            json={
                "schema_version": 1,
                "status": "ok",
                "tests": [
                    {"id": "test_solution.py::test_a", "outcome": "passed"},
                    {"id": "test_solution.py::test_b", "outcome": "failed"},
                ],
                "passed": 1,
                "failed": 1,
                "errors": 0,
                "stdout_excerpt": "x" * 500,
                "stderr_excerpt": "",
                "duration_ms": 12,
            },
        )

    report = await _backend(handler).execute(_request())
    assert report.status == "ok"
    assert report.passed == 1 and report.failed == 1
    assert [t.outcome for t in report.tests] == ["passed", "failed"]
    # A stdout bomb cannot grow downstream prompts unbounded.
    assert len(report.stdout_excerpt) <= 64 + len("\n[truncated]")


async def test_invalid_report_body_degrades():
    def handler(request):
        return httpx.Response(200, json={"status": "definitely-not-a-status"})

    report = await _backend(handler).execute(_request())
    assert report.status == "unavailable"


def test_extract_python_block_boundaries():
    fenced = "intro\n```python\ndef f():\n    return 1\n```\ntail"
    assert extract_python_block(fenced) == "def f():\n    return 1"
    assert extract_python_block("plain prose answer, nothing runnable") is None
    assert extract_python_block("```js\nconsole.log(1)\n```") is None
    assert extract_python_block("NOT_APPLICABLE") is None
    # Bare Python without a fence still counts iff it parses.
    assert extract_python_block("def g():\n    return 2") == "def g():\n    return 2"


def test_extract_python_block_salvages_truncated_fence():
    # A token-cap cutoff leaves an unclosed fence ending mid-statement; the
    # complete leading statements are salvaged as advisory evidence.
    truncated = (
        "```python\n"
        "def test_a():\n"
        "    assert f(1) == 2\n\n"
        "def test_b():\n"
        "    with pytest.raises(ValueError):\n"
    )
    salvaged = extract_python_block(truncated)
    assert salvaged is not None
    assert "def test_a" in salvaged
    assert "with pytest.raises" not in salvaged
    # An unclosed fence with no parseable prefix stays unextractable.
    assert extract_python_block("```python\ndef broken(:\n") is None
