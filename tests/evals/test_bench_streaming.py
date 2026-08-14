"""Target-only TTFT/TPS evidence and report aggregation."""

import json

import pytest

from evals.aggregate import build_scoreboard, render_markdown
from evals.streaming import ChatSSEAccumulator, StreamingProtocolError
from evals.types import GenerationTimingEvidence, ItemResult, PairResult


def _line(payload: dict) -> str:
    return "data: " + json.dumps(payload, separators=(",", ":"))


def test_stream_reconstruction_measures_first_semantic_delta_and_tps():
    stream = ChatSSEAccumulator(request_attempts=2)
    stream.feed_line(
        _line({"choices": [{"index": 0, "delta": {"role": "assistant"}}]}),
        0.02,
    )
    stream.feed_line(
        _line({"choices": [{"index": 0, "delta": {"content": "hello"}}]}),
        0.10,
    )
    stream.feed_line(
        _line(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": " world"},
                        "finish_reason": "stop",
                    }
                ]
            }
        ),
        0.30,
    )
    stream.feed_line(
        _line(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 3,
                    "total_tokens": 7,
                },
            }
        ),
        0.31,
    )
    stream.feed_line("data: [DONE]", 0.32)

    result = stream.finish()

    assert result.content == "hello world"
    assert result.timing is not None
    assert result.timing.ttft_s == 0.10
    assert result.timing.generation_span_s == pytest.approx(0.20)
    assert result.timing.tps == pytest.approx(10.0)
    assert result.timing.request_attempts == 2


def test_reasoning_and_tool_fragments_are_semantic_but_usage_is_not():
    stream = ChatSSEAccumulator(request_attempts=1)
    stream.feed_line(
        _line({"choices": [{"index": 0, "delta": {"reasoning_content": "think"}}]}),
        0.05,
    )
    stream.feed_line(
        _line(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "lookup", "arguments": "{}"},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        ),
        0.15,
    )
    stream.feed_line("data: [DONE]", 0.16)

    result = stream.finish()

    assert result.timing is not None
    assert result.timing.ttft_s == 0.05
    assert result.timing.tps is None
    assert result.timing.usage_missing is True
    assert json.loads(result.message_auxiliary or "{}")["tool_calls"][0]["id"] == "call_1"


def test_stream_requires_done_and_finish_reason():
    stream = ChatSSEAccumulator(request_attempts=1)
    stream.feed_line(
        _line({"choices": [{"index": 0, "delta": {"content": "x"}}]}),
        0.1,
    )

    with pytest.raises(StreamingProtocolError, match=r"without \[DONE\]"):
        stream.finish()


def test_stream_accepts_32k_completion_sized_sse_framing():
    stream = ChatSSEAccumulator(request_attempts=1)
    # The payload is deliberately larger than the old 1 MiB cap while the
    # semantic completion is tiny. Real 32K-token streams reach this size from
    # thousands of repeated JSON envelopes rather than one padding field.
    stream.feed_line(
        _line(
            {
                "padding": "x" * 1_100_000,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            }
        ),
        0.1,
    )
    stream.feed_line("data: [DONE]", 0.2)

    assert stream.finish().content == "ok"


def test_scoreboard_reports_target_timing_without_frontier_performance():
    timing = GenerationTimingEvidence(
        ttft_s=0.1,
        generation_span_s=0.2,
        completion_tokens=3,
        tps=10.0,
        semantic_events=2,
        request_attempts=1,
        usage_missing=False,
    )
    pair = PairResult(
        benchmark="gsm8k",
        target="m",
        status="completed",
        metrics={
            "score": 1.0,
            "n_total": 1,
            "n_scored": 1,
            "n_unjudged": 0,
            "n_skipped": 0,
            "n_failed": 0,
        },
        items=(
            ItemResult(
                item_id="1",
                status="completed",
                score=1.0,
                latency_s=0.4,
                timing=timing,
            ),
        ),
    )
    board = build_scoreboard(
        run_id="timing",
        suite="core",
        config={"attempts": 1},
        environment={},
        pairs=[pair],
        targets=["m"],
    )

    performance = board["cells"]["gsm8k"]["m"]["performance"]
    assert performance["ttft_p50_ms"] == 100.0
    assert performance["ttft_p95_ms"] == 100.0
    assert performance["tps_p50"] == 10.0
    markdown = render_markdown(board)
    assert "## Target generation performance" in markdown
    assert "no published-model performance is compared" in markdown


def test_loglikelihood_performance_is_explicitly_not_applicable():
    pair = PairResult(
        benchmark="mmlu",
        target="m",
        status="completed",
        metrics={"score": 1.0, "n_total": 0, "n_scored": 0},
    )
    board = build_scoreboard(
        run_id="mmlu-timing",
        suite="core",
        config={"attempts": 1},
        environment={},
        pairs=[pair],
        targets=["m"],
    )

    performance = board["cells"]["mmlu"]["m"]["performance"]
    assert performance["status"] == "not_applicable"
    assert "teacher-forced" in performance["reason"]
