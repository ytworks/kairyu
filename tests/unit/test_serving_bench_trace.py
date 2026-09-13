from __future__ import annotations

import json

import httpx
import pytest

from verification.l1.performance import serving_bench


@pytest.mark.parametrize(
    ("reason", "terminal", "complete"),
    [("stop", "data: [DONE]\n\n", True),
     ("length", "data: [DONE]\n\n", True),
     ("stop", "", False),
     ("stop", "data: [DONE]\n\ndata: [DONE]\n\n", False)],
)
async def test_stream_completion_evidence_keeps_finish_reason_without_saving_text(
    reason, terminal, complete
):
    chunks = [
        {"choices": [{"index": 0, "delta": {"content": "private fixture text"}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]},
        {"choices": [], "usage": {"completion_tokens": 3}},
    ]
    wire = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + terminal
    observed = []

    def upstream(request):
        payload = json.loads(request.content)
        assert payload["temperature"] == 1.0
        assert payload["top_p"] == 1.0
        return httpx.Response(200, text=wire)

    async with httpx.AsyncClient(
        base_url="http://bench.test/v1/",
        transport=httpx.MockTransport(upstream),
    ) as client:
        result = await serving_bench.run_one(
            client, "model", "prompt", 100, temperature=1.0, top_p=1.0,
            on_chunk=observed.append,
        )

    assert result.stream_complete is complete
    assert result.finish_reasons == (reason,)
    assert result.completion_tokens == 3
    assert result.response_text == ""
    assert observed == chunks


@pytest.mark.parametrize("finish_both", [False, True])
async def test_stream_completion_requires_a_terminal_for_each_requested_choice(finish_both):
    chunks = [{"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}]
    if finish_both:
        chunks.append({"choices": [{"index": 1, "delta": {}, "finish_reason": "stop"}]})
    wire = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"

    def upstream(request):
        body = json.loads(request.content)
        assert body["n"] == 2
        assert body["reasoning_effort"] == "max"
        return httpx.Response(200, text=wire)

    async with httpx.AsyncClient(
        base_url="http://bench.test/v1/", transport=httpx.MockTransport(upstream)
    ) as client:
        result = await serving_bench.run_one(client, "model", "prompt", 100,
                                            n=2, reasoning_effort="max")

    assert result.stream_complete is finish_both


def test_trace_retains_choice_verdict_without_private_text():
    trace = _execution_trace("ok")
    trace["events"][0].update(
        node="check", role="verifier", kind="verification",
        detail={"choice_index": 1, "pass": False, "inconclusive": False,
                "refinement_exhausted": True, "private_text": "must not persist"},
    )
    _version, stages = serving_bench._parse_trace(trace, response_id="req-1")
    row = stages[0].as_dict()
    assert row["choice_index"] == 1
    assert row["verification_pass"] is False
    assert row["verification_inconclusive"] is False
    assert row["refinement_exhausted"] is True
    assert "must not persist" not in json.dumps(row)


def _execution_trace(execution_status: object) -> dict:
    return {
        "trace_version": "2.0",
        "request_id": "req-1",
        "started_at": "2026-08-15T00:00:00.000Z",
        "completed_at": "2026-08-15T00:00:01.000Z",
        "events": [
            {
                "seq": 1,
                "node": "exec_matrix",
                "role": "executor",
                "kind": "execution",
                "status": "success",
                "attempt": 0,
                "timing": {
                    "queued_at": "2026-08-15T00:00:00.100Z",
                    "started_at": "2026-08-15T00:00:00.100Z",
                    "first_token_at": None,
                    "completed_at": "2026-08-15T00:00:00.200Z",
                },
                "usage": None,
                "detail": {"execution_status": execution_status},
            }
        ],
    }


def test_parse_trace_retains_execution_status_in_artifact_stage() -> None:
    version, stages = serving_bench._parse_trace(
        _execution_trace("unavailable"),
        response_id="req-1",
    )

    assert version == "2.0"
    assert len(stages) == 1
    assert stages[0].execution_status == "unavailable"
    assert stages[0].as_dict()["execution_status"] == "unavailable"


@pytest.mark.parametrize(
    "execution_status",
    [None, "UNAVAILABLE", "ok,", "ok,unavailable,", "x" * 129],
)
def test_parse_trace_rejects_unsafe_execution_status(
    execution_status: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="execution trace must report a safe execution_status",
    ):
        serving_bench._parse_trace(
            _execution_trace(execution_status),
            response_id="req-1",
        )


def test_parse_trace_omits_skipped_event_without_timing() -> None:
    """A conditional role skipped on a text request traces `timing: null`;
    the envelope stays valid and only observed stages are retained."""

    trace = _execution_trace("ok")
    trace["events"] = [
        {
            "seq": 1,
            "node": "image_description",
            "role": "proposal",
            "kind": "generation",
            "status": "skipped",
            "attempt": 0,
            "timing": None,
            "usage": None,
            "detail": {"reason": "no_image", "requires": "image"},
        },
        {**trace["events"][0], "seq": 2},
    ]

    version, stages = serving_bench._parse_trace(trace, response_id="req-1")

    assert version == "2.0"
    assert [stage.node for stage in stages] == ["exec_matrix"]


def test_parse_trace_still_rejects_successful_event_without_timing() -> None:
    trace = _execution_trace("ok")
    trace["events"][0]["timing"] = None

    with pytest.raises(ValueError, match="event.timing must be an object"):
        serving_bench._parse_trace(trace, response_id="req-1")
