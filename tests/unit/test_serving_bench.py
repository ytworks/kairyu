"""m9 D5: serving_bench honesty — token-granularity TPOT via include_usage."""

import asyncio
import datetime as dt
import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

_spec = importlib.util.spec_from_file_location(
    "serving_bench", Path(__file__).parents[2] / "bench" / "serving_bench.py"
)
serving_bench = importlib.util.module_from_spec(_spec)
sys.modules["serving_bench"] = serving_bench  # dataclass annotation resolution
_spec.loader.exec_module(serving_bench)


async def test_run_one_reads_usage_chunk_for_token_tpot():
    from fastapi import FastAPI, Header
    from fastapi.responses import StreamingResponse

    app = FastAPI()
    calls = []
    trace_headers = []

    @app.post("/v1/chat/completions")
    async def chat(
        request: dict,
        x_kairyu_trace: str | None = Header(default=None),
    ):
        calls.append(request)
        trace_headers.append(x_kairyu_trace)

        async def _gen():
            yield 'data: {"choices": [{"index": 0, "delta": {"content": "4"}}]}\n\n'
            yield (
                'data: {"choices": [], '
                '"usage": {"prompt_tokens": 3, "completion_tokens": 6, '
                '"total_tokens": 9}}\n\n'
            )
            yield "data: [DONE]\n\n"

        return StreamingResponse(_gen(), media_type="text/event-stream")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://t/v1"
    ) as client:
        metrics = await serving_bench.run_one(
            client,
            "m",
            "bench me please",
            max_tokens=6,
            temperature=0.0,
            seed=7,
            min_tokens=6,
            ignore_eos=True,
            capture_response=True,
        )
    assert calls[0]["stream_options"] == {"include_usage": True}
    assert trace_headers == [None]
    assert calls[0]["temperature"] == 0.0
    assert calls[0]["seed"] == 7
    assert calls[0]["min_tokens"] == 6
    assert calls[0]["ignore_eos"] is True
    assert metrics.completion_tokens == 6  # from the include_usage final chunk
    assert metrics.response_text == "4"
    assert metrics.token_granular is True
    assert metrics.trace_status == "not_requested"
    assert metrics.tpot_s >= 0.0


async def test_run_one_falls_back_when_target_rejects_stream_options():
    from fastapi import FastAPI, Header
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI()
    calls = []
    trace_headers = []

    @app.post("/v1/chat/completions")
    async def chat(
        request: dict,
        x_kairyu_trace: str | None = Header(default=None),
    ):
        calls.append(request)
        trace_headers.append(x_kairyu_trace)
        if "stream_options" in request:
            return JSONResponse(status_code=400, content={"error": "no stream_options"})

        async def _gen():
            yield 'data: {"choices": [{"index": 0, "delta": {"content": "hi"}}]}\n\n'
            yield "data: [DONE]\n\n"

        return StreamingResponse(_gen(), media_type="text/event-stream")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://t/v1"
    ) as client:
        metrics = await serving_bench.run_one(
            client,
            "m",
            "fallback",
            max_tokens=4,
            temperature=0.0,
            seed=11,
            min_tokens=4,
            ignore_eos=True,
            request_trace=True,
        )
    assert metrics.token_granular is False  # labeled chunk-granularity fallback
    assert metrics.trace_status == "missing"
    assert len(calls) == 2  # retried once without stream_options
    assert trace_headers == ["1", "1"]
    assert "stream_options" not in calls[1]
    for field, value in {
        "temperature": 0.0,
        "seed": 11,
        "min_tokens": 4,
        "ignore_eos": True,
    }.items():
        assert calls[1][field] == value


def test_summary_retains_raw_request_timings_and_token_throughput():
    results = [
        serving_bench.RequestMetrics(
            ttft_s=0.1,
            total_s=0.4,
            output_chunks=3,
            completion_tokens=4,
        ),
        serving_bench.RequestMetrics(
            ttft_s=0.2,
            total_s=0.5,
            output_chunks=4,
            completion_tokens=5,
        ),
    ]

    summary, samples = serving_bench.summarize_results(
        results,
        wall_ns=1_000_000_000,
        dataset_label="test",
        ttft_slo_s=0.15,
    )

    assert summary["wall_ns"] == 1_000_000_000
    assert summary["wall_s"] == 1.0
    assert summary["ttft_p50_ms"] == 100.0
    assert summary["completion_tokens_total"] == 9
    assert summary["percentile_method"] == "nearest-rank-v1"
    assert summary["output_tokens_per_s"] == 9.0
    assert summary["tpot_method"] == "token"
    assert summary["goodput_rps"] == 1.0
    assert summary["trace_coverage"] == {
        "total_requests": 2,
        "requested": 0,
            "not_requested": 2,
            "valid": 0,
            "partial": 0,
            "missing": 0,
        "invalid": 0,
        "fraction": None,
    }
    assert summary["stage_latency_ms"] == {}
    assert samples == [
        {
            "request_index": 0,
            "ttft_ms": 100.0,
            "total_ms": 400.0,
            "tpot_ms": pytest.approx(100.0),
            "output_chunks": 3,
            "completion_tokens": 4,
            "trace": {"status": "not_requested", "version": None, "stages": []},
        },
        {
            "request_index": 1,
            "ttft_ms": 200.0,
            "total_ms": 500.0,
            "tpot_ms": pytest.approx(75.0),
            "output_chunks": 4,
            "completion_tokens": 5,
            "trace": {"status": "not_requested", "version": None, "stages": []},
        },
    ]


def test_summary_separates_public_tokens_from_orchestration_usage():
    results = [
        serving_bench.RequestMetrics(
            ttft_s=0.1,
            total_s=0.5,
            output_chunks=3,
            completion_tokens=40,
            public_completion_tokens=5,
        ),
        serving_bench.RequestMetrics(
            ttft_s=0.2,
            total_s=0.6,
            output_chunks=3,
            completion_tokens=60,
            public_completion_tokens=7,
        ),
    ]

    summary, samples = serving_bench.summarize_results(
        results,
        wall_ns=1_000_000_000,
        dataset_label="orchestration",
        ttft_slo_s=1.0,
    )

    assert summary["completion_tokens_total"] == 100
    assert summary["output_tokens_per_s"] == 100.0
    assert summary["public_completion_tokens_total"] == 12
    assert summary["public_output_tokens_per_s"] == 12.0
    assert summary["tpot_method"] == "public-token"
    assert summary["generation_tokens_per_s_p50"] == 12.5
    assert summary["public_generation_tokens_per_s_mean"] == 12.5
    assert summary["reported_output_tokens_per_e2e_s_mean"] == 90.0
    assert summary["e2e_p50_ms"] == 500.0
    assert summary["e2e_p99_ms"] == 600.0
    assert [sample["public_completion_tokens"] for sample in samples] == [5, 7]
    assert "response_text" not in str(samples)


def test_shared_target_form_normalizes_url_and_uses_env_secret(monkeypatch):
    monkeypatch.setenv("SERVING_API_KEY", "private-value")
    args = serving_bench.build_parser().parse_args(
        [
            "--target",
            "gateway=http://gateway.test:8000/=model=SERVING_API_KEY",
        ]
    )

    target = serving_bench.resolve_target(args)
    config = serving_bench.build_run_config(args)

    assert target.base_url == "http://gateway.test:8000/v1"
    assert serving_bench.resolve_api_key(args, target) == "private-value"
    assert config["api_key_env"] == "SERVING_API_KEY"
    assert config["api_key_source"] == "environment"
    assert config["stage_trace"] is False
    assert config["profile"] is False
    assert config["concurrency"] == 16
    assert "private-value" not in str(config)


def test_stage_trace_is_an_explicit_recorded_opt_in():
    args = serving_bench.build_parser().parse_args(["--stage-trace"])

    assert args.stage_trace is True
    assert serving_bench.build_run_config(args)["stage_trace"] is True


def test_public_tokenizer_requires_a_pinned_model_and_records_both():
    parser = serving_bench.build_parser()
    incomplete = parser.parse_args(
        ["--public-tokenizer-url", "http://tokenizer.test/tokenize"]
    )
    with pytest.raises(ValueError, match="provided together"):
        serving_bench.serving_artifact_paths(incomplete)

    args = parser.parse_args(
        [
            "--public-tokenizer-url",
            "http://tokenizer.test/tokenize",
            "--public-tokenizer-model",
            "pinned-model",
            "--results-dir",
            "",
        ]
    )
    config = serving_bench.build_run_config(args)
    assert config["public_tokenizer_url"] == "http://tokenizer.test/tokenize"
    assert config["public_tokenizer_model"] == "pinned-model"


def test_client_profile_is_an_independent_recorded_opt_in():
    args = serving_bench.build_parser().parse_args(["--profile", "--stage-trace"])
    config = serving_bench.build_run_config(args)

    assert args.profile is True
    assert config["profile"] is True
    assert config["stage_trace"] is True
    help_text = serving_bench.build_parser().format_help()
    assert "local" in help_text
    assert "benchmark client" in help_text


def test_client_profile_requires_results_before_target_or_network_work():
    args = serving_bench.build_parser().parse_args(
        ["--profile", "--results-dir", ""]
    )

    with pytest.raises(ValueError, match="--profile.*--results-dir"):
        serving_bench.serving_artifact_paths(args)


def test_client_profile_uses_one_utc_microsecond_result_stem(tmp_path: Path):
    args = serving_bench.build_parser().parse_args(
        ["--profile", "--results-dir", str(tmp_path)]
    )
    result, trace = serving_bench.serving_artifact_paths(
        args,
        now=dt.datetime(2026, 8, 6, 2, 3, 4, 5678, tzinfo=dt.UTC),
    )

    assert result == tmp_path / "2026-08-06T020304.005678Z-serving.json"
    assert trace == (
        tmp_path / "2026-08-06T020304.005678Z-serving.client.pt.trace.json"
    )


@pytest.mark.parametrize(
    "arguments",
    [
        ["--num-requests", "0"],
        ["--concurrency", "0"],
        ["--max-tokens", "0"],
        ["--tensor-parallel", "0"],
        ["--dp-replicas", "0"],
        ["--ttft-slo-s", "nan"],
        ["--ttft-slo-s", "inf"],
        ["--ttft-slo-s", "0"],
        ["--timeout", "nan"],
        ["--timeout", "-1"],
        ["--temperature", "nan"],
        ["--temperature", "inf"],
        ["--temperature", "-1"],
        ["--min-tokens", "-1"],
        ["--max-tokens", "4", "--min-tokens", "5"],
    ],
)
def test_invalid_run_arguments_fail_before_artifact_creation(
    tmp_path: Path,
    arguments: list[str],
):
    results_dir = tmp_path / "results"
    args = serving_bench.build_parser().parse_args(
        [*arguments, "--results-dir", str(results_dir)]
    )

    with pytest.raises(ValueError):
        serving_bench.serving_artifact_paths(args)

    assert not results_dir.exists()


def test_shared_target_refuses_conflicting_legacy_endpoint_flags():
    args = serving_bench.build_parser().parse_args(
        [
            "--target",
            "gateway=http://gateway.test=model",
            "--model",
            "other-model",
        ]
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        serving_bench.resolve_target(args)


def test_missing_explicit_api_key_environment_fails_closed(monkeypatch):
    monkeypatch.delenv("MISSING_SERVING_API_KEY", raising=False)
    args = serving_bench.build_parser().parse_args(
        ["--api-key-env", "MISSING_SERVING_API_KEY"]
    )
    target = serving_bench.resolve_target(args)

    with pytest.raises(ValueError, match="MISSING_SERVING_API_KEY"):
        serving_bench.resolve_api_key(args, target)


def test_legacy_literal_api_key_never_enters_recorded_config():
    secret = "literal-secret-value"
    args = serving_bench.build_parser().parse_args(["--api-key", secret])
    target = serving_bench.resolve_target(args)
    config = serving_bench.build_run_config(args)

    assert serving_bench.resolve_api_key(args, target) == secret
    assert config["api_key_source"] == "legacy-cli"
    assert secret not in str(config)
    assert "Deprecated literal bearer token" in serving_bench.build_parser().format_help()


def test_serving_percentile_uses_nearest_rank_definition():
    assert serving_bench._percentile(list(range(1, 21)), 0.95) == 19


def _trace(*, request_id="chatcmpl-traced"):
    return {
        "trace_version": "2.0",
        "request_id": request_id,
        "started_at": "2026-08-05T00:00:00.000Z",
        "completed_at": "2026-08-05T00:00:00.100Z",
        "events": [
            {
                "seq": 1,
                "node": "router",
                "role": "router",
                "kind": "routing",
                "status": "success",
                "attempt": 0,
                "timing": {
                    "queued_at": None,
                    "started_at": "2026-08-05T00:00:00.000Z",
                    "first_token_at": None,
                    "completed_at": "2026-08-05T00:00:00.005Z",
                },
                "detail": {"secret": "must-not-be-retained"},
            },
            {
                "seq": 2,
                "node": "worker",
                "role": "direct",
                "kind": "generation",
                "status": "success",
                "attempt": 1,
                "timing": {
                    "queued_at": "2026-08-05T00:00:00.010Z",
                    "started_at": "2026-08-05T00:00:00.020Z",
                    "first_token_at": "2026-08-05T00:00:00.050Z",
                    "completed_at": "2026-08-05T00:00:00.100Z",
                },
                "detail": {"prompt": "also-must-not-be-retained"},
            },
        ],
    }


def test_trace_retains_only_bounded_moa_count_and_usage_scalars():
    trace = _trace()
    trace["events"][1].update(
        {
            "node": "moa",
            "role": "moa",
            "kind": "synthesis",
            "attempt": 0,
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 45,
                "cached_tokens": 20,
            },
            "detail": {"proposals": 3, "secret": "must-not-be-retained"},
        }
    )

    _, stages = serving_bench._parse_trace(trace, response_id="chatcmpl-traced")
    moa = stages[1]
    assert moa.proposals == 3
    assert (moa.input_tokens, moa.output_tokens, moa.cached_tokens) == (120, 45, 20)
    assert "secret" not in str(moa.as_dict())

    metric = serving_bench.RequestMetrics(
        ttft_s=0.1,
        total_s=0.2,
        output_chunks=2,
        completion_tokens=3,
        trace_status="valid",
        trace_version="2.0",
        trace_stages=stages,
    )
    summary, _ = serving_bench.summarize_results(
        [metric],
        wall_ns=1_000_000_000,
        dataset_label="moa-trace",
        ttft_slo_s=1.0,
    )
    moa_summary = summary["stage_latency_ms"]["moa/moa/synthesis"]
    assert moa_summary["proposal_counts"] == [3]
    assert moa_summary["usage_totals"] == {
        "input_tokens": 120,
        "output_tokens": 45,
        "cached_tokens": 20,
    }


@pytest.mark.parametrize("proposals", [None, 0, 17, "3"])
def test_successful_moa_trace_rejects_missing_or_unsafe_proposal_count(proposals):
    trace = _trace()
    trace["events"][1].update(
        {
            "node": "moa",
            "role": "moa",
            "kind": "synthesis",
            "detail": {"proposals": proposals},
        }
    )

    with pytest.raises(ValueError, match="1..16 proposals"):
        serving_bench._parse_trace(trace, response_id="chatcmpl-traced")


async def test_run_one_extracts_only_sanitized_terminal_trace_and_not_as_output():
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat():
        trace = _trace()

        async def _gen():
            yield (
                'data: {"id":"chatcmpl-traced","choices":'
                '[{"index":0,"delta":{"content":"hi"}}]}\n\n'
            )
            yield "data: " + json.dumps(
                {
                    "id": "chatcmpl-traced",
                    "choices": [],
                    "usage": {"completion_tokens": 3},
                    "kairyu_trace_v2": trace,
                }
            ) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_gen(), media_type="text/event-stream")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t/v1"
    ) as client:
        unrequested = await serving_bench.run_one(client, "m", "prompt", 3)
        metrics = await serving_bench.run_one(
            client, "m", "secret prompt", 3, request_trace=True
        )

    assert unrequested.trace_status == "not_requested"
    assert unrequested.trace_stages == ()
    assert metrics.output_chunks == 1
    assert metrics.completion_tokens == 3
    assert metrics.trace_status == "valid"
    assert metrics.trace_version == "2.0"
    assert [stage.stage for stage in metrics.trace_stages] == [
        "router/router/routing",
        "worker/direct/generation",
    ]
    worker = metrics.trace_stages[1]
    assert worker.status == "success"
    assert worker.attempt == 1
    assert worker.queue_wait_ms == pytest.approx(10.0)
    assert worker.service_ms == pytest.approx(80.0)
    assert worker.first_token_ms == pytest.approx(30.0)
    assert worker.post_first_ms == pytest.approx(50.0)
    assert worker.total_ms == pytest.approx(90.0)
    assert "secret" not in str(worker.as_dict())
    assert "prompt" not in str(worker.as_dict())


async def test_run_one_isolates_invalid_terminal_trace_from_latency_results():
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat():
        trace = _trace(request_id="wrong-id")

        async def _gen():
            yield (
                'data: {"id":"chatcmpl-traced","choices":'
                '[{"index":0,"delta":{"content":"hi"}}]}\n\n'
            )
            yield "data: " + json.dumps(
                {
                    "id": "chatcmpl-traced",
                    "choices": [],
                    "kairyu_trace_v2": trace,
                }
            ) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_gen(), media_type="text/event-stream")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t/v1"
    ) as client:
        metrics = await serving_bench.run_one(
            client, "m", "prompt", 3, request_trace=True
        )

    assert metrics.output_chunks == 1
    assert metrics.trace_status == "invalid"
    assert metrics.trace_version is None
    assert metrics.trace_stages == ()


@pytest.mark.parametrize(
    ("mode", "expected_status"),
    (
        ("empty", "missing"),
        ("misplaced", "invalid"),
        ("after-terminal", "invalid"),
        ("no-done", "invalid"),
    ),
)
async def test_run_one_requires_nonempty_trace_in_final_event_before_done(
    mode,
    expected_status,
):
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat():
        trace = _trace()
        if mode == "empty":
            trace["events"] = []

        async def _gen():
            content = {
                "id": "chatcmpl-traced",
                "choices": [{"index": 0, "delta": {"content": "hi"}}],
            }
            if mode == "misplaced":
                content["kairyu_trace_v2"] = {"untrusted": "not-terminal"}
            yield f"data: {json.dumps(content)}\n\n"
            yield "data: " + json.dumps(
                {
                    "id": "chatcmpl-traced",
                    "choices": [],
                    "kairyu_trace_v2": trace,
                }
            ) + "\n\n"
            if mode == "after-terminal":
                yield (
                    'data: {"id":"chatcmpl-traced","choices":[]}\n\n'
                )
            if mode != "no-done":
                yield "data: [DONE]\n\n"

        return StreamingResponse(_gen(), media_type="text/event-stream")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t/v1"
    ) as client:
        metrics = await serving_bench.run_one(
            client, "m", "prompt", 3, request_trace=True
        )

    assert metrics.trace_status == expected_status
    if mode == "empty":
        assert metrics.trace_version == "2.0"
    else:
        assert metrics.trace_version is None
    assert metrics.trace_stages == ()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda trace: trace.__setitem__("trace_version", "3.0"),
        lambda trace: trace.__setitem__("request_id", "other"),
        lambda trace: trace["events"][0].__setitem__("seq", 2),
        lambda trace: trace.__setitem__("started_at", "2026-08-05T00:00:00"),
        lambda trace: trace["events"][1]["timing"].__setitem__(
            "completed_at", "2026-08-05T00:00:00.001Z"
        ),
    ],
    ids=["version", "request-id", "seq", "non-utc", "reversed"],
)
def test_trace_validation_rejects_malformed_evidence(mutate):
    trace = _trace()
    mutate(trace)

    with pytest.raises(ValueError):
        serving_bench._parse_trace(trace, response_id="chatcmpl-traced")


def test_trace_ignores_unknown_kind_and_status_without_retaining_identities():
    trace = _trace()
    trace["events"].append(
        {
            "seq": 3,
            "node": "secret prompt must not persist",
            "role": "sk-live-secret",
            "kind": "future_scalar_event",
            "status": "future",
            "attempt": 0,
            "timing": None,
            "detail": {"secret": "must-not-be-retained"},
        }
    )
    trace["events"][1]["status"] = "future-status"

    _, stages = serving_bench._parse_trace(
        trace, response_id="chatcmpl-traced"
    )

    assert [stage.stage for stage in stages] == ["router/router/routing"]
    assert "secret" not in str([stage.as_dict() for stage in stages])


@pytest.mark.parametrize("field", ["node", "role"])
def test_trace_omits_unsafe_persisted_identity_without_invalidating_envelope(field):
    trace = _trace()
    trace["events"][0][field] = "secret prompt canary"

    version, stages = serving_bench._parse_trace(
        trace, response_id="chatcmpl-traced"
    )

    assert version == "2.0"
    assert [stage.stage for stage in stages] == ["worker/direct/generation"]
    assert "secret prompt canary" not in str(
        [stage.as_dict() for stage in stages]
    )


def _direct_stage_trace():
    return {
        "trace_version": "2.0",
        "request_id": "chatcmpl-stage",
        "started_at": "2026-08-05T00:00:00.000Z",
        "completed_at": "2026-08-05T00:00:00.100Z",
        "events": [
            {
                "seq": 1,
                "node": "tokenize",
                "role": "engine",
                "kind": "stage",
                "status": "success",
                "attempt": 0,
                "timing": None,
                "detail": {
                    "stage": "tokenize",
                    "duration_ns": 1_500_000,
                    "occurrences": 2,
                    "aggregation": "sum",
                    "scope": "request-observed",
                },
            }
        ],
    }


async def test_run_one_classifies_complete_direct_stage_trace_as_valid():
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse

    trace = _direct_stage_trace()
    template = trace["events"][0]
    stage_names = (
        "tokenize",
        "queue_wait",
        "schedule",
        "prefill",
        "decode_step",
        "detokenize",
        "sse_write",
        "future_stage",
    )
    trace["events"] = [
        {
            **template,
            "seq": index,
            "node": stage,
            "detail": {
                **template["detail"],
                "stage": stage,
                "duration_ns": index * 1_000_000,
            },
        }
        for index, stage in enumerate(stage_names, start=1)
    ]
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat():
        async def _gen():
            yield (
                'data: {"id":"chatcmpl-stage","choices":'
                '[{"index":0,"delta":{"content":"hi"}}]}\n\n'
            )
            yield "data: " + json.dumps(
                {
                    "id": "chatcmpl-stage",
                    "choices": [],
                    "kairyu_trace_v2": trace,
                }
            ) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_gen(), media_type="text/event-stream")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t/v1"
    ) as client:
        metrics = await serving_bench.run_one(
            client, "m", "prompt", 3, request_trace=True
        )

    assert metrics.trace_status == "valid"
    assert [stage.stage for stage in metrics.trace_stages] == list(stage_names)


def test_direct_stage_trace_uses_scalar_duration_and_discards_raw_detail():
    version, stages = serving_bench._parse_trace(
        _direct_stage_trace(), response_id="chatcmpl-stage"
    )

    assert version == "2.0"
    assert len(stages) == 1
    stage = stages[0]
    assert stage.stage == "tokenize"
    assert stage.duration_ms == 1.5
    assert stage.occurrences == 2
    assert stage.queue_wait_ms is None
    assert stage.service_ms is None
    assert set(stage.as_dict()) == {
        "stage",
        "node",
        "role",
        "kind",
        "status",
        "attempt",
        "duration_ms",
        "occurrences",
        "queue_wait_ms",
        "service_ms",
        "first_token_ms",
        "post_first_ms",
        "total_ms",
        "input_tokens",
        "output_tokens",
        "cached_tokens",
        "proposals",
    }
    assert "aggregation" not in stage.as_dict()
    assert "scope" not in stage.as_dict()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda event: event["detail"].__setitem__("duration_ns", True),
        lambda event: event["detail"].__setitem__("duration_ns", -1),
        lambda event: event["detail"].__setitem__("duration_ns", 1 << 63),
        lambda event: event["detail"].__setitem__("occurrences", True),
        lambda event: event["detail"].__setitem__("occurrences", 0),
        lambda event: event["detail"].__setitem__("occurrences", 1 << 31),
        lambda event: event["detail"].__setitem__("stage", "../tokenize"),
        lambda event: event.__setitem__("node", "other"),
        lambda event: event["detail"].__setitem__("aggregation", "mean"),
        lambda event: event["detail"].__setitem__("scope", "global"),
        lambda event: event.__setitem__("timing", {}),
    ],
    ids=[
        "duration-bool",
        "duration-negative",
        "duration-overflow",
        "occurrences-bool",
        "occurrences-zero",
        "occurrences-overflow",
        "unsafe-stage",
        "node-mismatch",
        "aggregation",
        "scope",
        "timing",
    ],
)
def test_direct_stage_trace_rejects_untrusted_scalar_detail(mutate):
    trace = _direct_stage_trace()
    mutate(trace["events"][0])

    with pytest.raises(ValueError):
        serving_bench._parse_trace(trace, response_id="chatcmpl-stage")


def test_direct_stage_trace_ignores_additive_detail_without_retaining_it():
    trace = _direct_stage_trace()
    trace["events"][0]["detail"]["future_field"] = "do-not-retain"

    _, stages = serving_bench._parse_trace(
        trace, response_id="chatcmpl-stage"
    )

    assert stages[0].stage == "tokenize"
    assert "future_field" not in stages[0].as_dict()
    assert "do-not-retain" not in str(stages[0].as_dict())


def test_direct_stage_summary_reports_duration_percentiles_and_occurrence_total():
    _, first = serving_bench._parse_trace(
        _direct_stage_trace(), response_id="chatcmpl-stage"
    )
    second_trace = _direct_stage_trace()
    second_trace["events"][0]["detail"]["duration_ns"] = 3_000_000
    second_trace["events"][0]["detail"]["occurrences"] = 4
    _, second = serving_bench._parse_trace(
        second_trace, response_id="chatcmpl-stage"
    )
    common = dict(
        ttft_s=0.1,
        total_s=0.2,
        output_chunks=2,
        completion_tokens=3,
        trace_status="partial",
        trace_version="2.0",
    )

    summary, _ = serving_bench.summarize_results(
        [
            serving_bench.RequestMetrics(**common, trace_stages=first),
            serving_bench.RequestMetrics(**common, trace_stages=second),
        ],
        wall_ns=1_000_000_000,
        dataset_label="stage-trace",
        ttft_slo_s=1.0,
    )

    stage = summary["stage_latency_ms"]["tokenize"]
    assert stage["duration"] == {"samples": 2, "p50_ms": 1.5, "p99_ms": 3.0}
    assert stage["occurrences_total"] == 6
    assert stage["expected_native"] is True
    assert stage["requests_observed"] == 2
    assert stage["requests_missing"] == 0
    assert stage["coverage_fraction"] == 1.0
    assert stage["service"] == {"samples": 0, "p50_ms": None, "p99_ms": None}
    missing = summary["stage_latency_ms"]["decode_step"]
    assert missing["requests_observed"] == 0
    assert missing["requests_missing"] == 2
    assert missing["coverage_fraction"] == 0.0
    assert missing["duration"] == {
        "samples": 0,
        "p50_ms": None,
        "p99_ms": None,
    }
    assert summary["trace_coverage"]["partial"] == 2


def test_summary_aggregates_stage_percentiles_and_keeps_missing_invalid_distinct():
    _, base_stages = serving_bench._parse_trace(
        _trace(), response_id="chatcmpl-traced"
    )
    slower = tuple(
        serving_bench.TraceStageMetrics(
            **{
                **stage.__dict__,
                "queue_wait_ms": (
                    stage.queue_wait_ms * 2 if stage.queue_wait_ms is not None else None
                ),
                "service_ms": stage.service_ms * 2,
                "first_token_ms": (
                    stage.first_token_ms * 2 if stage.first_token_ms is not None else None
                ),
                "post_first_ms": (
                    stage.post_first_ms * 2 if stage.post_first_ms is not None else None
                ),
                "total_ms": stage.total_ms * 2,
            }
        )
        for stage in base_stages
    )
    base = dict(ttft_s=0.1, total_s=0.2, output_chunks=2, completion_tokens=3)
    results = [
        serving_bench.RequestMetrics(
            **base,
            trace_status="valid",
            trace_version="2.0",
            trace_stages=base_stages,
        ),
        serving_bench.RequestMetrics(
            **base,
            trace_status="valid",
            trace_version="2.0",
            trace_stages=slower,
        ),
        serving_bench.RequestMetrics(**base, trace_status="missing"),
        serving_bench.RequestMetrics(**base, trace_status="invalid"),
        serving_bench.RequestMetrics(**base),
    ]

    summary, samples = serving_bench.summarize_results(
        results,
        wall_ns=1_000_000_000,
        dataset_label="trace-test",
        ttft_slo_s=1.0,
    )

    assert summary["trace_coverage"] == {
        "total_requests": 5,
        "requested": 4,
            "not_requested": 1,
            "valid": 2,
            "partial": 0,
            "missing": 1,
        "invalid": 1,
        "fraction": 0.5,
    }
    worker = summary["stage_latency_ms"]["worker/direct/generation"]
    assert worker["queue_wait"] == {"samples": 2, "p50_ms": 10.0, "p99_ms": 20.0}
    assert worker["service"] == {"samples": 2, "p50_ms": 80.0, "p99_ms": 160.0}
    router = summary["stage_latency_ms"]["router/router/routing"]
    assert router["queue_wait"] == {"samples": 0, "p50_ms": None, "p99_ms": None}
    assert samples[2]["trace"] == {"status": "missing", "version": None, "stages": []}
    assert samples[3]["trace"] == {"status": "invalid", "version": None, "stages": []}
    assert samples[4]["trace"] == {
        "status": "not_requested",
        "version": None,
        "stages": [],
    }


def test_profiled_run_writes_bound_client_trace_without_secrets(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    events: list[str] = []
    captured_client: dict[str, object] = {}
    secret = "profile-api-key-canary"
    prompt = "profile-prompt-canary"

    class FakeProfiler:
        def export_chrome_trace(self, path: str) -> None:
            events.append("export")
            Path(path).write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            {
                                "name": "kairyu.bench.serving.client-measurement",
                                "ph": "X",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

    @contextmanager
    def fake_profile_scope(**kwargs):
        assert kwargs == {
            "enabled": True,
            "activities": ("cpu",),
            "acc_events": True,
            "record_shapes": False,
            "profile_memory": False,
            "with_stack": False,
        }
        events.append("profile-enter")
        try:
            yield FakeProfiler()
        finally:
            events.append("profile-exit")

    @contextmanager
    def fake_record_function(enabled: bool, name: str):
        assert enabled is True
        assert name == "kairyu.bench.serving.client-measurement"
        events.append("range-enter")
        try:
            yield
        finally:
            events.append("range-exit")

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            captured_client.update(kwargs)

        async def __aenter__(self):
            events.append("client-enter")
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            del exc_type, exc, traceback
            events.append("client-exit")

    async def fake_run_one(*args, **kwargs):
        del args, kwargs
        events.append("request")
        return serving_bench.RequestMetrics(
            ttft_s=0.01,
            total_s=0.02,
            output_chunks=2,
            completion_tokens=3,
        )

    ticks = iter((1_000_000_000, 2_000_000_000))
    monkeypatch.setattr(serving_bench, "profile_scope", fake_profile_scope)
    monkeypatch.setattr(
        serving_bench,
        "record_function_scope",
        fake_record_function,
    )
    monkeypatch.setattr(serving_bench.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(serving_bench, "run_one", fake_run_one)
    monkeypatch.setattr(
        serving_bench,
        "load_prompts",
        lambda dataset, count: ([prompt], "synthetic"),
    )
    monkeypatch.setattr(serving_bench.time, "perf_counter_ns", lambda: next(ticks))
    args = serving_bench.build_parser().parse_args(
        [
            "--profile",
            "--api-key",
            secret,
            "--num-requests",
            "1",
            "--concurrency",
            "1",
            "--results-dir",
            str(tmp_path),
        ]
    )

    asyncio.run(serving_bench.run_benchmark(args))

    result_paths = list(tmp_path.glob("*-serving.json"))
    trace_paths = list(tmp_path.glob("*-serving.client.pt.trace.json"))
    assert len(result_paths) == 1
    assert len(trace_paths) == 1
    result = json.loads(result_paths[0].read_text(encoding="utf-8"))
    trace_bytes = trace_paths[0].read_bytes()
    metadata = result["artifacts"]["torch_profile"]
    assert metadata["path"] == trace_paths[0].name
    assert metadata["scope"] == "local-client-process"
    assert metadata["activities"] == ["cpu"]
    assert metadata["target_process_included"] is False
    assert metadata["diagnostic_only"] is True
    assert metadata["timing_comparable"] is False
    assert metadata["size_bytes"] == len(trace_bytes)
    assert result["config"]["profile"] is True
    assert captured_client["headers"] == {"Authorization": f"Bearer {secret}"}
    assert events == [
        "client-enter",
        "profile-enter",
        "range-enter",
        "request",
        "range-exit",
        "profile-exit",
        "client-exit",
        "export",
    ]
    retained = result_paths[0].read_text(encoding="utf-8") + trace_bytes.decode()
    assert secret not in retained
    assert prompt not in retained
    console = capsys.readouterr()
    assert secret not in console.out + console.err
    assert "excludes the target server" in console.err


def test_profile_export_failure_does_not_publish_success_result(
    tmp_path: Path,
    monkeypatch,
):
    @contextmanager
    def fake_profile_scope(**kwargs):
        assert kwargs["enabled"] is True
        yield object()

    @contextmanager
    def fake_record_function(enabled: bool, name: str):
        assert enabled is True
        assert name == "kairyu.bench.serving.client-measurement"
        yield

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            del kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            del exc_type, exc, traceback

    async def fake_run_one(*args, **kwargs):
        del args, kwargs
        return serving_bench.RequestMetrics(
            ttft_s=0.01,
            total_s=0.02,
            output_chunks=2,
            completion_tokens=3,
        )

    ticks = iter((1_000_000_000, 2_000_000_000))
    monkeypatch.setattr(serving_bench, "profile_scope", fake_profile_scope)
    monkeypatch.setattr(
        serving_bench,
        "record_function_scope",
        fake_record_function,
    )
    monkeypatch.setattr(serving_bench.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(serving_bench, "run_one", fake_run_one)
    monkeypatch.setattr(
        serving_bench,
        "load_prompts",
        lambda dataset, count: (["prompt"], "synthetic"),
    )
    monkeypatch.setattr(serving_bench.time, "perf_counter_ns", lambda: next(ticks))

    def fail_export(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("profile export failed")

    monkeypatch.setattr(serving_bench, "write_trace_artifact", fail_export)
    args = serving_bench.build_parser().parse_args(
        [
            "--profile",
            "--num-requests",
            "1",
            "--results-dir",
            str(tmp_path),
        ]
    )

    with pytest.raises(RuntimeError, match="profile export failed"):
        asyncio.run(serving_bench.run_benchmark(args))

    assert list(tmp_path.iterdir()) == []


def test_profiled_result_publish_race_preserves_winner_and_rolls_back_trace(
    tmp_path: Path,
    monkeypatch,
):
    class FakeProfiler:
        def export_chrome_trace(self, path: str) -> None:
            Path(path).write_text(
                json.dumps({"traceEvents": [{"name": "client", "ph": "X"}]}),
                encoding="utf-8",
            )

    @contextmanager
    def fake_profile_scope(**kwargs):
        assert kwargs["enabled"] is True
        yield FakeProfiler()

    @contextmanager
    def fake_record_function(enabled: bool, name: str):
        assert enabled is True
        assert name == "kairyu.bench.serving.client-measurement"
        yield

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            del kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            del exc_type, exc, traceback

    async def fake_run_one(*args, **kwargs):
        del args, kwargs
        return serving_bench.RequestMetrics(
            ttft_s=0.01,
            total_s=0.02,
            output_chunks=2,
            completion_tokens=3,
        )

    ticks = iter((1_000_000_000, 2_000_000_000))
    monkeypatch.setattr(serving_bench, "profile_scope", fake_profile_scope)
    monkeypatch.setattr(serving_bench, "record_function_scope", fake_record_function)
    monkeypatch.setattr(serving_bench.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(serving_bench, "run_one", fake_run_one)
    monkeypatch.setattr(
        serving_bench,
        "load_prompts",
        lambda dataset, count: (["prompt"], "synthetic"),
    )
    monkeypatch.setattr(serving_bench.time, "perf_counter_ns", lambda: next(ticks))
    real_atomic_write_json = serving_bench.atomic_write_json

    def lose_result_publication_race(path, payload, **kwargs):
        Path(path).write_text("race winner", encoding="utf-8")
        return real_atomic_write_json(path, payload, **kwargs)

    monkeypatch.setattr(
        serving_bench,
        "atomic_write_json",
        lose_result_publication_race,
    )
    args = serving_bench.build_parser().parse_args(
        [
            "--profile",
            "--num-requests",
            "1",
            "--concurrency",
            "1",
            "--results-dir",
            str(tmp_path),
        ]
    )

    with pytest.raises(FileExistsError):
        asyncio.run(serving_bench.run_benchmark(args))

    retained = list(tmp_path.iterdir())
    assert len(retained) == 1
    assert retained[0].name.endswith("-serving.json")
    assert retained[0].read_text(encoding="utf-8") == "race winner"
