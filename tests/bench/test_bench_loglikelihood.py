"""Fail-closed teacher-forced continuation scoring and MMLU integration."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from conftest import make_target
from pydantic import ValidationError

from kairyu.bench.adapters.base import (
    RequestFailed,
    RunContext,
    call_loglikelihood,
)
from kairyu.bench.adapters.mmlu import MmluAdapter
from kairyu.bench.cache import BenchCache
from kairyu.bench.types import (
    BenchItem,
    ContinuationLogLikelihood,
    LogLikelihoodRequestSpec,
    LogLikelihoodResponse,
)


def _choice_payload(
    continuation: str,
    logprobs: tuple[float, ...] = (-0.25,),
    *,
    token_ids: tuple[int, ...] | None = None,
) -> dict:
    token_base = 10
    if continuation in {" A", " B", " C", " D"}:
        token_base += ord(continuation[-1]) - ord("A")
    ids = token_ids or tuple(range(token_base, token_base + len(logprobs)))
    return {
        "mode": "loglikelihood",
        "choices": [
            {
                "index": 0,
                "text": continuation,
                "finish_reason": "length",
                "prompt_token_ids": [101, 102, 103],
                "continuation_token_ids": list(ids),
                "logprobs": {
                    "tokens": [f"token-{index}" for index in range(len(logprobs))],
                    "token_logprobs": list(logprobs),
                    "text_offset": list(range(len(logprobs))),
                },
            }
        ],
    }


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _ctx(tmp_path, handler, **overrides) -> RunContext:
    defaults = dict(
        cache=BenchCache(tmp_path / "cache"),
        http_factory=lambda: _client(handler),
        offline_fixtures=True,
        retries=0,
    )
    defaults.update(overrides)
    return RunContext(**defaults)


def _candidate(letter: str, value: float, *, token_count: int = 1):
    values = (value,) * token_count
    token_base = 10 + ord(letter) - ord("A")
    return ContinuationLogLikelihood(
        continuation=f" {letter}",
        token_ids=tuple(range(token_base, token_base + token_count)),
        tokens=tuple(f"token-{index}" for index in range(token_count)),
        token_logprobs=values,
        text_offsets=tuple(range(token_count)),
        sum_logprob=sum(values),
        score=sum(values),
    )


def _mmlu_response(
    values: tuple[float, float, float, float], *, token_count: int = 1
) -> LogLikelihoodResponse:
    return LogLikelihoodResponse(
        reduction="sum",
        prompt_token_ids=(101, 102, 103),
        candidates=tuple(
            _candidate(letter, value, token_count=token_count)
            for letter, value in zip("ABCD", values, strict=True)
        ),
    )


@pytest.mark.parametrize(
    "continuations",
    [(), ("",), (" A", " A")],
)
def test_loglikelihood_request_requires_nonempty_unique_candidates(
    continuations,
) -> None:
    with pytest.raises(ValidationError):
        LogLikelihoodRequestSpec(context="Q", continuations=continuations)


def test_loglikelihood_request_requires_nonempty_context() -> None:
    with pytest.raises(ValidationError, match="context must be non-empty"):
        LogLikelihoodRequestSpec(context="", continuations=(" A",))


async def test_loglikelihood_wire_is_exact_ordered_authenticated_and_retried(
    monkeypatch,
) -> None:
    seen: list[tuple[str, dict, str | None]] = []
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        body = json.loads(request.content)
        seen.append((request.url.path, body, request.headers.get("authorization")))
        if attempts == 0:
            attempts += 1
            return httpx.Response(429, text="busy")
        return httpx.Response(
            200,
            json=_choice_payload(body["kairyu_continuation"]),
        )

    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    target = make_target(
        reasoning_effort="high",
        top_p=0.7,
        seed=99,
        extra_body_json='{"vendor_knob": true}',
    )
    request = LogLikelihoodRequestSpec(
        context="Question\nAnswer:",
        continuations=(" A", " B"),
    )
    async with _client(handler) as client:
        response = await call_loglikelihood(
            client,
            target,
            request,
            retries=1,
            timeout_s=3.0,
            api_key="secret",
        )

    assert sleeps == [0.5]
    assert [entry[1]["kairyu_continuation"] for entry in seen] == [" A", " A", " B"]
    assert all(path == "/v1/completions" for path, _, _ in seen)
    assert all(auth == "Bearer secret" for _, _, auth in seen)
    assert all(
        set(body)
        == {
            "model",
            "prompt",
            "kairyu_continuation",
            "max_tokens",
            "temperature",
            "stream",
            "logprobs",
        }
        for _, body, _ in seen
    )
    assert all(body["max_tokens"] is None for _, body, _ in seen)
    assert all(body["temperature"] == 0.0 for _, body, _ in seen)
    assert all(body["stream"] is False for _, body, _ in seen)
    assert all(body["logprobs"] == 0 for _, body, _ in seen)
    assert [candidate.continuation for candidate in response.candidates] == [" A", " B"]


@pytest.mark.parametrize("retry_status", [500, 502, 503, 504])
async def test_loglikelihood_retries_every_retryable_server_status(
    retry_status, monkeypatch
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(retry_status, text="retry")
        body = json.loads(request.content)
        return httpx.Response(200, json=_choice_payload(body["kairyu_continuation"]))

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    async with _client(handler) as client:
        await call_loglikelihood(
            client,
            make_target(),
            LogLikelihoodRequestSpec(context="Q", continuations=(" A",)),
            retries=1,
            timeout_s=1,
        )
    assert calls == 2


async def test_loglikelihood_retries_transport_errors(monkeypatch) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("disconnected", request=request)
        body = json.loads(request.content)
        return httpx.Response(200, json=_choice_payload(body["kairyu_continuation"]))

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    async with _client(handler) as client:
        await call_loglikelihood(
            client,
            make_target(),
            LogLikelihoodRequestSpec(context="Q", continuations=(" A",)),
            retries=1,
            timeout_s=1,
        )
    assert calls == 2


def _malformed_payload(case: str) -> object:
    payload = _choice_payload(" A")
    choice = payload["choices"][0]
    if case == "top-level":
        return []
    if case == "mode":
        payload.pop("mode")
    elif case == "choices":
        payload["choices"].append(choice.copy())
    elif case == "text":
        choice["text"] = " B"
    elif case == "index":
        choice["index"] = 1
    elif case == "finish-reason":
        choice["finish_reason"] = "stop"
    elif case == "prompt-token-ids":
        choice.pop("prompt_token_ids")
    elif case == "token-ids":
        choice["continuation_token_ids"] = []
    elif case == "logprobs":
        choice.pop("logprobs")
    elif case == "tokens":
        choice["logprobs"].pop("tokens")
    elif case == "token-logprobs":
        choice["logprobs"]["token_logprobs"] = []
    elif case == "offsets":
        choice["logprobs"]["text_offset"] = []
    return payload


@pytest.mark.parametrize(
    "case",
    [
        "top-level",
        "mode",
        "choices",
        "text",
        "index",
        "finish-reason",
        "prompt-token-ids",
        "token-ids",
        "logprobs",
        "tokens",
        "token-logprobs",
        "offsets",
    ],
)
async def test_loglikelihood_rejects_malformed_or_missing_evidence(case) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_malformed_payload(case))

    async with _client(handler) as client:
        with pytest.raises(RequestFailed, match="loglikelihood"):
            await call_loglikelihood(
                client,
                make_target(),
                LogLikelihoodRequestSpec(context="Q", continuations=(" A",)),
                retries=0,
                timeout_s=1,
            )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 0.01])
async def test_loglikelihood_rejects_nonfinite_or_positive_logprob(value) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=json.dumps(
                _choice_payload(" A", (value,)), allow_nan=True
            ).encode(),
            headers={"content-type": "application/json"},
        )

    async with _client(handler) as client:
        with pytest.raises(RequestFailed, match="finite values <= 0"):
            await call_loglikelihood(
                client,
                make_target(),
                LogLikelihoodRequestSpec(context="Q", continuations=(" A",)),
                retries=0,
                timeout_s=1,
            )


async def test_loglikelihood_sum_and_mean_token_reductions() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json=_choice_payload(body["kairyu_continuation"], (-1.0, -3.0)),
        )

    async with _client(handler) as client:
        summed = await call_loglikelihood(
            client,
            make_target(),
            LogLikelihoodRequestSpec(
                context="Q", continuations=(" answer",), reduction="sum"
            ),
            retries=0,
            timeout_s=1,
        )
        mean = await call_loglikelihood(
            client,
            make_target(),
            LogLikelihoodRequestSpec(
                context="Q", continuations=(" answer",), reduction="mean_token"
            ),
            retries=0,
            timeout_s=1,
        )
    assert summed.candidates[0].sum_logprob == -4.0
    assert summed.candidates[0].score == -4.0
    assert mean.candidates[0].sum_logprob == -4.0
    assert mean.candidates[0].score == -2.0


async def test_loglikelihood_requires_same_prompt_tokens_for_every_candidate() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        body = json.loads(request.content)
        payload = _choice_payload(body["kairyu_continuation"])
        if calls == 2:
            payload["choices"][0]["prompt_token_ids"] = [999]
        return httpx.Response(200, json=payload)

    async with _client(handler) as client:
        with pytest.raises(RequestFailed, match="prompt_token_ids changed"):
            await call_loglikelihood(
                client,
                make_target(),
                LogLikelihoodRequestSpec(
                    context="Q", continuations=(" A", " B")
                ),
                retries=0,
                timeout_s=1,
            )


async def test_mmlu_tie_uses_stable_candidate_order_and_records_evidence(
    tmp_path,
) -> None:
    adapter = MmluAdapter()
    item = BenchItem(
        id="tie",
        payload={
            "question": "Q?",
            "subject": "formal_logic",
            "choices": ["one", "two", "three", "four"],
            "answer": "B",
        },
    )
    result = await adapter.score(
        item,
        _mmlu_response((-0.1, -0.1, -2.0, -3.0)),
        _ctx(tmp_path, lambda _: httpx.Response(500)),
    )
    assert result.status == "completed"
    assert result.score == 0.0
    assert result.details["winner"] == "A"
    assert result.details["gold"] == "B"
    assert result.details["ties"] == ["A", "B"]
    assert result.details["margin"] == 0.0
    assert result.details["reduction"] == "sum"
    assert result.details["candidates"][0]["label"] == "A"
    assert result.details["candidates"][0]["text"] == "one"


async def test_mmlu_multitoken_candidate_is_failed_and_unmeasured(tmp_path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json=_choice_payload(
                body["kairyu_continuation"],
                (-0.1, -0.2),
                token_ids=(1, 2),
            ),
        )

    ctx = _ctx(tmp_path, handler)
    pair = await MmluAdapter().run(make_target(), ctx)
    assert pair.status == "failed"
    assert pair.metrics["score"] is None
    assert pair.metrics["n_scored"] == 0
    assert pair.metrics["n_failed"] == 2
    assert pair.items[0].score is None
    assert "exactly one" in pair.items[0].error
    assert "target tokenizer" in pair.reason
    assert "not attempted after fatal capability probe" in pair.items[1].error
    assert calls == 4


async def test_mmlu_duplicate_single_token_ids_are_failed_and_unmeasured(
    tmp_path,
) -> None:
    item = BenchItem(
        id="duplicate-token",
        payload={
            "question": "Q?",
            "subject": "formal_logic",
            "choices": ["one", "two", "three", "four"],
            "answer": "A",
        },
    )
    response = LogLikelihoodResponse(
        reduction="sum",
        prompt_token_ids=(101,),
        candidates=tuple(
            ContinuationLogLikelihood(
                continuation=f" {letter}",
                token_ids=(7,),
                tokens=(f" {letter}",),
                token_logprobs=(value,),
                text_offsets=(0,),
                sum_logprob=value,
                score=value,
            )
            for letter, value in zip("ABCD", (-0.1, -1.0, -2.0, -3.0), strict=True)
        ),
    )
    result = await MmluAdapter().score(
        item,
        response,
        _ctx(tmp_path, lambda _: httpx.Response(500)),
    )
    assert result.status == "failed"
    assert result.score is None
    assert "four distinct" in result.error


async def test_mmlu_e2e_ranks_known_candidates_with_custom_transport(tmp_path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        continuation = body["kairyu_continuation"]
        calls.append(continuation)
        gold = " B" if "only even prime" in body["prompt"] else " D"
        rank = {" A": -4.0, " B": -3.0, " C": -2.0, " D": -1.0}
        value = -0.01 if continuation == gold else rank[continuation]
        return httpx.Response(200, json=_choice_payload(continuation, (value,)))

    pair = await MmluAdapter().run(make_target(), _ctx(tmp_path, handler))
    assert pair.status == "completed"
    assert pair.metrics["score"] == 1.0
    assert pair.metrics["n_scored"] == 2
    assert [item.details["winner"] for item in pair.items] == ["B", "D"]
    assert calls == [" A", " B", " C", " D", " A", " B", " C", " D"]


async def test_explicitly_unsupported_extension_skips_pair_before_fanout(
    tmp_path,
) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            400,
            json={"error": "kairyu_continuation is unsupported by this target"},
        )

    pair = await MmluAdapter().run(make_target(), _ctx(tmp_path, handler))
    assert pair.status == "skipped"
    assert pair.metrics["score"] is None
    assert "does not support" in pair.reason
    assert calls == 1


async def test_silent_ordinary_completion_skips_pair_before_fanout(tmp_path) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={"choices": [{"index": 0, "text": "ordinary completion"}]},
        )

    pair = await MmluAdapter().run(make_target(), _ctx(tmp_path, handler))
    assert pair.status == "skipped"
    assert "does not support" in pair.reason
    assert calls == 1


async def test_explicit_422_unknown_field_skips_pair_before_fanout(tmp_path) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            422,
            json={
                "detail": [
                    {
                        "loc": ["body", "kairyu_continuation"],
                        "type": "extra_forbidden",
                        "msg": "Extra inputs are not permitted",
                    }
                ]
            },
        )

    pair = await MmluAdapter().run(make_target(), _ctx(tmp_path, handler))
    assert pair.status == "skipped"
    assert calls == 1


@pytest.mark.parametrize(
    ("status_code", "body"),
    [
        (400, {"error": "Unknown parameter: 'kairyu_continuation'."}),
        (400, {"error": "unknown field kairyu_continuation"}),
        (422, {"error": "unrecognized field kairyu_continuation"}),
        (
            400,
            {
                "error": (
                    "Unrecognized request argument supplied: "
                    "kairyu_continuation"
                )
            },
        ),
    ],
)
async def test_explicit_unknown_extension_wording_skips_before_fanout(
    tmp_path, status_code, body
) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status_code, json=body)

    pair = await MmluAdapter().run(make_target(), _ctx(tmp_path, handler))
    assert pair.status == "skipped"
    assert pair.metrics["score"] is None
    assert calls == 1


@pytest.mark.parametrize(
    ("status_code", "body", "reason"),
    [
        (401, {"error": "invalid token"}, "authentication"),
        (403, {"error": "forbidden"}, "authentication"),
        (404, {"error": {"code": "model_not_found"}}, "not found"),
        (404, {"detail": "route missing"}, "not found"),
    ],
)
async def test_fatal_probe_errors_fail_pair_without_fanout(
    tmp_path, status_code, body, reason
) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status_code, json=body)

    pair = await MmluAdapter().run(make_target(), _ctx(tmp_path, handler))
    assert pair.status == "failed"
    assert reason in pair.reason
    assert pair.metrics["score"] is None
    assert calls == 1


@pytest.mark.parametrize("status_code", [429, 502])
async def test_exhausted_retryable_probe_failure_stops_fanout(
    tmp_path, monkeypatch, status_code
) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status_code, text="still unavailable")

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    pair = await MmluAdapter().run(
        make_target(),
        _ctx(tmp_path, handler, retries=2),
    )

    assert pair.status == "failed"
    assert "unavailable" in pair.reason
    assert pair.metrics["score"] is None
    assert calls == 3


async def test_exhausted_transport_probe_failure_stops_fanout(
    tmp_path, monkeypatch
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("offline", request=request)

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    pair = await MmluAdapter().run(
        make_target(),
        _ctx(tmp_path, handler, retries=2),
    )

    assert pair.status == "failed"
    assert "transport" in pair.reason
    assert pair.metrics["score"] is None
    assert calls == 3


async def test_generic_400_stays_failed_item_evidence(tmp_path) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"error": "bad request"})

    pair = await MmluAdapter().run(make_target(), _ctx(tmp_path, handler))
    assert pair.status == "failed"
    assert pair.metrics["score"] is None
    assert pair.metrics["n_failed"] == 2
    assert calls == 2


async def test_token_boundary_probe_failure_stops_before_remaining_items(
    tmp_path,
) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            400,
            json={
                "error": (
                    "kairyu_continuation does not align with a model token boundary"
                )
            },
        )

    pair = await MmluAdapter().run(make_target(), _ctx(tmp_path, handler))
    assert pair.status == "failed"
    assert pair.metrics["score"] is None
    assert pair.metrics["n_failed"] == 2
    assert "fixed continuation boundary" in pair.reason
    assert "HTTP 400" in pair.items[0].error
    assert "not attempted after fatal capability probe" in pair.items[1].error
    # The boundary is rejected on the probe's first ordered candidate.  No
    # other candidate or dataset item is sent.
    assert calls == 1


async def test_malformed_evidence_becomes_failed_unmeasured_item(tmp_path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        body = json.loads(request.content)
        payload = _choice_payload(body["kairyu_continuation"])
        payload["choices"][0]["logprobs"].pop("token_logprobs")
        return httpx.Response(200, json=payload)

    pair = await MmluAdapter().run(make_target(), _ctx(tmp_path, handler))
    assert pair.status == "failed"
    assert pair.metrics["score"] is None
    assert pair.metrics["n_failed"] == 2
    assert pair.items[0].status == "failed"
    assert pair.items[0].score is None
    assert "invalid loglikelihood response" in pair.items[0].error
    assert "malformed log-likelihood evidence" in pair.reason
    assert "not attempted after fatal capability probe" in pair.items[1].error
    assert calls == 1
