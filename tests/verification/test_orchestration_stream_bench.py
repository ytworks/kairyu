import json

import httpx

from verification.orchestration.correctness.orchestration_stream_bench import (
    RequestMeasurement,
    measure_request,
    run_transport_ab,
    summarize_requests,
)


def test_request_summary_uses_measured_content_ttft():
    measurements = [
        RequestMeasurement(
            model="m",
            order=index % 2,
            prompt_sha256=str(index),
            ttft_ms=float(value),
            total_ms=float(value + 10),
            content_chunks=2,
            completion_tokens=2,
            status_comments=1,
            trace_comments=1,
        )
        for index, value in enumerate((1, 2, 3, 4))
    ]

    summary = summarize_requests(measurements)

    assert summary["requests"] == 4
    assert summary["ttft_p50_ms"] == 2.5
    assert summary["ttft_p99_ms"] == 4
    assert summary["completion_tokens_min"] == 2
    assert summary["completion_tokens_max"] == 2
    assert summary["status_comments"] == 4
    assert summary["trace_comments"] == 4


async def test_measure_request_ignores_comments_and_records_usage():
    content_chunk = {
        "choices": [{"delta": {"role": "assistant", "content": "hello"}}]
    }
    usage_chunk = {
        "choices": [],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }
    body = (
        ": status working\n\n"
        f"data: {json.dumps(content_chunk)}\n\n"
        f"data: {json.dumps(usage_chunk)}\n\n"
        ": trace route: tier1\n\n"
        "data: [DONE]\n\n"
    )

    async def handler(request):
        assert request.headers["x-kairyu-trace"] == "1"
        return httpx.Response(200, content=body)

    async with httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await measure_request(
            client,
            model="auto",
            prompt="prompt",
            order=1,
            max_tokens=8,
        )

    assert result.content_chunks == 1
    assert result.completion_tokens == 1
    assert result.status_comments == 1
    assert result.trace_comments == 1
    assert result.ttft_ms <= result.total_ms


async def test_transport_ab_consumes_every_event_in_both_candidates():
    result = await run_transport_ab(chunks=1_000, rounds=3)

    assert result["chunks_per_round"] == 1_000
    assert result["rounds"] == 3
    assert set(result["median_ns_per_event"]) == {
        "pull_through",
        "queue_bridge",
    }
    assert result["selected"] in result["median_ns_per_event"]
    assert result["queue_over_pull_through"] > 0
