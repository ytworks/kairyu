import asyncio
import gc
import threading
import time
import weakref
from collections.abc import Mapping

import httpx
import pytest

import kairyu.engine.kairyu_backend as kairyu_backend_module
from kairyu import SamplingParams
from kairyu.engine.backend import (
    GenerationRequest,
    GenerationResult,
    GenerationStageMetric,
    GenerationUsage,
)
from kairyu.engine.core.sampling_types import SampledToken
from kairyu.engine.core.scheduler import ScheduledChunk
from kairyu.engine.engine_loop import StreamUpdate
from kairyu.engine.kairyu_backend import KairyuBackend, build_engine_loop
from kairyu.engine.prompt import MultimodalItem, MultimodalPrompt, TokensPrompt
from kairyu.engine.registry import create_backend
from kairyu.engine.tokenizer import ToyTokenizer
from kairyu.entrypoints.server.app import create_app
from kairyu.outputs import CompletionOutput, TokenLogprob


def _request(request_id: str, prompt: str, max_tokens: int = 4) -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        prompt=prompt,
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"stage": "", "duration_ns": 0}, "safe identifier"),
        ({"stage": "../prefill", "duration_ns": 0}, "safe identifier"),
        ({"stage": "x" * 65, "duration_ns": 0}, "safe identifier"),
        ({"stage": "prefill", "duration_ns": -1}, "non-negative integer"),
        ({"stage": "prefill", "duration_ns": True}, "non-negative integer"),
        (
            {"stage": "prefill", "duration_ns": 1 << 63},
            "non-negative integer",
        ),
        (
            {"stage": "prefill", "duration_ns": 0, "occurrences": 0},
            "positive integer",
        ),
        (
            {"stage": "prefill", "duration_ns": 0, "occurrences": True},
            "positive integer",
        ),
        (
            {"stage": "prefill", "duration_ns": 0, "occurrences": 1 << 31},
            "positive integer",
        ),
    ),
)
def test_generation_stage_metric_validates_strict_scalar_contract(kwargs, message):
    with pytest.raises(ValueError, match=message):
        GenerationStageMetric(**kwargs)


def test_generation_trace_fields_preserve_historical_constructor_order() -> None:
    params = SamplingParams(max_tokens=1)
    request = GenerationRequest("request", "prompt", params)
    completion = CompletionOutput(index=0, text="done", token_ids=(7,))
    usage = GenerationUsage(prompt_tokens=1, completion_tokens=1)
    result = GenerationResult(
        "request",
        "prompt",
        (completion,),
        True,
        usage,
        (1,),
    )

    assert request.trace_requested is False
    assert result.prompt_token_ids == (1,)
    assert result.stage_metrics == ()
    with pytest.raises(ValueError, match="trace_requested must be a boolean"):
        GenerationRequest("bad", "prompt", params, trace_requested=1)  # type: ignore[arg-type]


class _SlowRunner:
    def execute(
        self,
        scheduled: tuple[ScheduledChunk, ...],
        states: Mapping[str, object],
    ) -> dict[str, tuple[SampledToken, ...]]:
        time.sleep(0.01)
        return {
            chunk.request_id: (SampledToken(7),)
            for chunk in scheduled
            if not chunk.is_prefill or states[chunk.request_id].prefill_done
        }


class _StepGateRunner:
    def __init__(self, gated_call: int) -> None:
        self.gated_call = gated_call
        self.calls = 0
        self.entered = threading.Event()
        self.allow_step = threading.Event()

    def execute(
        self,
        scheduled: tuple[ScheduledChunk, ...],
        states: Mapping[str, object],
    ) -> dict[str, tuple[SampledToken, ...]]:
        self.calls += 1
        if self.calls == self.gated_call:
            self.entered.set()
            if not self.allow_step.wait(timeout=2):
                raise TimeoutError("test did not release the gated engine step")
        return {
            chunk.request_id: (SampledToken(7),)
            for chunk in scheduled
            if not chunk.is_prefill or states[chunk.request_id].prefill_done
        }


class _FailOnceRunner:
    def __init__(self) -> None:
        self.failed = False

    def execute(
        self,
        scheduled: tuple[ScheduledChunk, ...],
        states: Mapping[str, object],
    ) -> dict[str, tuple[SampledToken, ...]]:
        if not self.failed:
            self.failed = True
            raise RuntimeError("injected runner failure")
        return {
            chunk.request_id: (SampledToken(7),)
            for chunk in scheduled
            if not chunk.is_prefill or states[chunk.request_id].prefill_done
        }


class _ToggleTokenizer:
    eos_token_id = None

    def __init__(self) -> None:
        self.return_token = False

    def encode(self, text: str) -> tuple[int, ...]:
        return (1,) if self.return_token else ()

    def decode(self, token_ids) -> str:
        return " ".join(f"tok{token_id}" for token_id in token_ids)

    def vocab(self) -> list[str]:
        return ["<empty>", "token"]


async def test_zero_token_prompt_is_rejected_before_backend_state():
    tokenizer = _ToggleTokenizer()
    backend = KairyuBackend(tokenizer=tokenizer)
    request = _request("empty", "normal-looking input")

    with pytest.raises(ValueError, match="at least one token"):
        backend.validate_request(request)
    with pytest.raises(ValueError, match="at least one token"):
        await backend.generate(request)

    assert backend._active_request_ids == set()
    assert backend._loop._active_request_ids == set()
    assert backend._queues == {}
    assert backend._scheduler.states == {}

    tokenizer.return_token = True
    result = await backend.generate(request)
    assert result.finished is True


async def test_context_limit_rejects_before_backend_state_and_allows_boundary_retry():
    backend = KairyuBackend(
        tokenizer=ToyTokenizer(),
        num_pages=64,
        max_model_len=5,
    )
    oversized = GenerationRequest(
        request_id="context",
        prompt=TokensPrompt((1, 2, 3, 4)),
        sampling_params=SamplingParams(max_tokens=2),
    )

    with pytest.raises(ValueError, match="exceed max_model_len"):
        backend.validate_request(oversized)
    with pytest.raises(ValueError, match="exceed max_model_len"):
        await backend.generate(oversized)

    assert backend._active_request_ids == set()
    assert backend._loop._active_request_ids == set()
    assert backend._queues == {}
    assert backend._scheduler.states == {}

    boundary = GenerationRequest(
        request_id="context",
        prompt=TokensPrompt((1, 2, 3)),
        sampling_params=SamplingParams(max_tokens=2),
    )
    result = await backend.generate(boundary)

    assert result.finished
    assert result.usage is not None
    assert result.usage.prompt_tokens + result.usage.completion_tokens == 5
    await backend.shutdown()


class _NoEncodeTokenizer(ToyTokenizer):
    def encode(self, text: str) -> tuple[int, ...]:
        raise AssertionError("pre-tokenized prompts must not call encode")


class _CountingTokenizer(ToyTokenizer):
    def __init__(self) -> None:
        self.encoded_texts: list[str] = []

    def encode(self, text: str) -> tuple[int, ...]:
        self.encoded_texts.append(text)
        return super().encode(text)


class _LoglikelihoodBoundaryTokenizer(ToyTokenizer):
    def encode(self, text: str) -> tuple[int, ...]:
        known = {
            "context": (1, 2),
            "context continuation": (1, 2, 7, 9),
        }
        if text in known:
            return known[text]
        return super().encode(text)


def test_native_backend_exposes_exact_loglikelihood_token_boundary():
    backend = KairyuBackend(
        tokenizer=_LoglikelihoodBoundaryTokenizer(),
        num_pages=64,
    )

    assert backend.tokenize_loglikelihood("context", " continuation") == (
        (1, 2),
        (7, 9),
    )

    backend._loop.close()


async def test_trace_metrics_survive_preflight_and_backend_result_boundary():
    tokenizer = _CountingTokenizer()
    backend = KairyuBackend(tokenizer=tokenizer, num_pages=64)
    request = GenerationRequest(
        request_id="traced-result",
        prompt="one two three",
        sampling_params=SamplingParams(max_tokens=3, ignore_eos=True),
        trace_requested=True,
    )

    backend.validate_request(request)
    result = await backend.generate(request)
    metrics = {metric.stage: metric for metric in result.stage_metrics}

    assert tokenizer.encoded_texts == ["one two three"]
    assert tuple(metric.stage for metric in result.stage_metrics) == (
        "tokenize",
        "queue_wait",
        "schedule",
        "prefill",
        "decode_step",
        "detokenize",
    )
    assert metrics["tokenize"].occurrences == 1
    assert metrics["queue_wait"].occurrences == 1
    assert all(metric.duration_ns >= 0 for metric in result.stage_metrics)
    assert all(metric.occurrences >= 1 for metric in result.stage_metrics)

    untraced = await backend.generate(_request("untraced-result", "plain", 1))
    assert untraced.stage_metrics == ()
    await backend.shutdown()


async def test_stream_stage_metrics_are_cumulative_and_stage_unique():
    backend = KairyuBackend(num_pages=64)
    request = GenerationRequest(
        request_id="traced-stream",
        prompt="one two three",
        sampling_params=SamplingParams(max_tokens=4, ignore_eos=True),
        trace_requested=True,
    )

    partials = [partial async for partial in backend.stream(request)]

    previous: dict[str, GenerationStageMetric] = {}
    for partial in partials:
        current = {metric.stage: metric for metric in partial.stage_metrics}
        assert len(current) == len(partial.stage_metrics)
        for stage, metric in current.items():
            if stage in previous:
                assert metric.duration_ns >= previous[stage].duration_ns
                assert metric.occurrences >= previous[stage].occurrences
        previous = current
    assert partials[-1].finished
    assert "decode_step" in previous
    await backend.shutdown()


@pytest.mark.parametrize("stream", [False, True])
async def test_n_candidates_attribute_shared_tokenization_once(stream):
    tokenizer = _CountingTokenizer()
    backend = KairyuBackend(tokenizer=tokenizer, num_pages=128)
    request = GenerationRequest(
        request_id=f"traced-n-{'stream' if stream else 'generate'}",
        prompt="one shared prompt",
        sampling_params=SamplingParams(
            max_tokens=2,
            n=2,
            temperature=0,
            ignore_eos=True,
        ),
        trace_requested=True,
    )

    backend.validate_request(request)
    if stream:
        result = [partial async for partial in backend.stream(request)][-1]
    else:
        result = await backend.generate(request)

    tokenize = next(
        metric for metric in result.stage_metrics if metric.stage == "tokenize"
    )
    assert tokenizer.encoded_texts == ["one shared prompt"]
    assert tokenize.occurrences == 1
    await backend.shutdown()


async def test_token_prompt_bypasses_encode_and_reports_exact_input_ids():
    backend = KairyuBackend(tokenizer=_NoEncodeTokenizer(), num_pages=64)
    request = GenerationRequest(
        request_id="tokens",
        prompt=TokensPrompt(prompt_token_ids=(7, 11, 13)),
        sampling_params=SamplingParams(max_tokens=2, temperature=0),
    )

    result = await backend.generate(request)

    assert result.prompt == request.prompt
    assert result.prompt_token_ids == (7, 11, 13)
    assert result.usage is not None
    assert result.usage.prompt_tokens == 3


async def test_validated_tool_request_reuses_exact_prepared_tokens_for_n_choices():
    tokenizer = _CountingTokenizer()
    backend = KairyuBackend(tokenizer=tokenizer, num_pages=256)
    request = GenerationRequest(
        request_id="prepared-tools",
        prompt="find the weather",
        sampling_params=SamplingParams(max_tokens=2, n=3, temperature=0),
        tools=(
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object"},
                },
            },
        ),
        tool_choice="required",
    )

    backend.validate_request(request)
    backend.validate_request(request)
    result = await backend.generate(request)

    assert len(tokenizer.encoded_texts) == 1
    assert "Available functions:" in tokenizer.encoded_texts[0]
    assert "You must call one of the available functions." in tokenizer.encoded_texts[0]
    assert len(result.completions) == 3
    assert all(completion.finish_reason == "length" for completion in result.completions)
    assert backend._prepared_requests == {}
    await backend.shutdown()


async def test_direct_generate_without_preflight_still_tokenizes_once():
    tokenizer = _CountingTokenizer()
    backend = KairyuBackend(tokenizer=tokenizer, num_pages=64)

    result = await backend.generate(_request("direct", "direct request", max_tokens=1))

    assert result.finished
    assert tokenizer.encoded_texts == ["direct request"]
    assert backend._prepared_requests == {}
    await backend.shutdown()


def test_abandoned_validated_request_releases_prepared_tokens():
    tokenizer = _CountingTokenizer()
    backend = KairyuBackend(tokenizer=tokenizer, num_pages=64)
    request = _request("abandoned-validation", "validate but never submit")
    request_ref = weakref.ref(request)

    backend.validate_request(request)
    assert len(backend._prepared_requests) == 1

    del request
    gc.collect()

    assert request_ref() is None
    assert backend._prepared_requests == {}
    backend._loop.close()


async def test_text_and_equivalent_token_prompt_share_native_radix_cache():
    tokenizer = ToyTokenizer()
    backend = KairyuBackend(tokenizer=tokenizer, num_pages=256, page_size=4)
    text = " ".join(f"shared-{index}" for index in range(20))
    token_ids = tokenizer.encode(text)
    params = SamplingParams(max_tokens=1, temperature=0)

    text_result = await backend.generate(
        GenerationRequest("cache-text", text, params)
    )
    token_result = await backend.generate(
        GenerationRequest(
            "cache-tokens",
            TokensPrompt(token_ids, prompt="display text is not a cache key"),
            params,
        )
    )

    assert text_result.completions[0].token_ids == token_result.completions[0].token_ids
    assert text_result.usage is not None
    assert token_result.usage is not None
    assert text_result.usage.cached_tokens == 0
    assert token_result.usage.prompt_tokens == len(token_ids)
    assert token_result.usage.cached_tokens > 0


def test_token_tool_suffix_and_multimodal_reject_before_backend_state():
    backend = KairyuBackend(num_pages=64)
    tool_request = GenerationRequest(
        request_id="token-tools",
        prompt=TokensPrompt((1, 2, 3)),
        sampling_params=SamplingParams(max_tokens=1),
        tools=(
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object"},
                },
            },
        ),
    )
    multimodal_request = GenerationRequest(
        request_id="multimodal",
        prompt=MultimodalPrompt(
            base="describe",
            items=(
                MultimodalItem(
                    modality="image",
                    encoding="uri",
                    data="https://example.test/image.png",
                ),
            ),
        ),
        sampling_params=SamplingParams(max_tokens=1),
    )

    with pytest.raises(ValueError, match="tool-instruction"):
        backend.validate_request(tool_request)
    with pytest.raises(ValueError, match="does not support multimodal"):
        backend.validate_request(multimodal_request)
    assert backend._active_request_ids == set()
    assert backend._loop._active_request_ids == set()
    assert backend._scheduler.states == {}


def test_token_id_outside_backend_vocabulary_rejects_before_state():
    backend = KairyuBackend(tokenizer=_ToggleTokenizer(), num_pages=64)
    request = GenerationRequest(
        request_id="out-of-vocab",
        prompt=TokensPrompt((2,)),
        sampling_params=SamplingParams(max_tokens=1),
    )

    with pytest.raises(ValueError, match="outside the backend tokenizer vocabulary"):
        backend.validate_request(request)
    assert backend._active_request_ids == set()
    assert backend._loop._active_request_ids == set()
    assert backend._scheduler.states == {}


@pytest.mark.parametrize(
    ("field", "params"),
    [
        ("best_of", SamplingParams(best_of=2)),
        ("prompt_logprobs", SamplingParams(prompt_logprobs=1)),
        ("extra_args.vendor", SamplingParams(extra_args={"vendor": True})),
    ],
)
def test_validate_request_rejects_native_unsupported_intent(field, params):
    backend = KairyuBackend()
    request = GenerationRequest(
        request_id="unsupported",
        prompt="hello",
        sampling_params=params,
    )

    with pytest.raises(ValueError, match=field):
        backend.validate_request(request)


def test_validate_request_accepts_explicit_special_token_output_policy():
    backend = KairyuBackend()
    request = GenerationRequest(
        request_id="keep-special-tokens",
        prompt="hello",
        sampling_params=SamplingParams(skip_special_tokens=False),
    )

    backend.validate_request(request)

    assert backend._peek_prepared_request(request) is not None
    assert backend._active_request_ids == set()
    assert backend._loop._active_request_ids == set()
    assert backend._scheduler.states == {}


def test_validate_request_rejects_native_strict_tool_semantics():
    backend = KairyuBackend()
    request = GenerationRequest(
        request_id="strict",
        prompt="hello",
        sampling_params=SamplingParams(),
        tools=(
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object"},
                    "strict": True,
                },
            },
        ),
    )

    with pytest.raises(ValueError, match=r"tools\[0\]\.function\.strict"):
        backend.validate_request(request)


async def test_generate_runs_through_engine_core():
    backend = KairyuBackend(num_pages=256)
    result = await backend.generate(_request("r1", "hello world from kairyu"))
    assert result.finished is True
    completion = result.completions[0]
    assert len(completion.token_ids) == 4  # max_tokens honored by the scheduler
    assert completion.text
    assert completion.finish_reason == "length"


async def test_stream_yields_incremental_partials():
    backend = KairyuBackend(num_pages=256)
    partials = []
    async for partial in backend.stream(_request("r2", "stream me", max_tokens=5)):
        partials.append(partial)
    assert partials
    assert partials[-1].finished is True
    assert all(p.finished is False for p in partials[:-1])
    lengths = [len(p.completions[0].token_ids) for p in partials]
    assert lengths == sorted(lengths)  # monotonically growing
    assert lengths[-1] == 5


async def test_stream_coalesces_backlogged_cumulative_updates_to_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = KairyuBackend(num_pages=64)
    queue: asyncio.Queue[StreamUpdate] = asyncio.Queue()
    monkeypatch.setattr(backend, "_submit", lambda _request: queue)
    stream = backend.stream(_request("backlogged", "prompt", max_tokens=4))

    queue.put_nowait(
        StreamUpdate(
            outputs=(7,),
            text="a",
            finished=False,
            finish_reason=None,
            num_prompt_tokens=3,
            num_cached_tokens=1,
        )
    )
    first = await anext(stream)

    queue.put_nowait(
        StreamUpdate(
            outputs=(7, 8),
            text="ab",
            finished=False,
            finish_reason=None,
            num_prompt_tokens=3,
            num_cached_tokens=1,
        )
    )
    queue.put_nowait(
        StreamUpdate(
            outputs=(7, 8, 9),
            text="abc",
            finished=False,
            finish_reason=None,
            num_prompt_tokens=3,
            num_cached_tokens=1,
        )
    )
    queue.put_nowait(
        StreamUpdate(
            outputs=(7, 8, 9, 10),
            text="abcd",
            finished=True,
            finish_reason="length",
            logprobs=({7: -0.1}, {8: -0.2}, {9: -0.3}, {10: -0.4}),
            cumulative_logprob=-1.0,
            num_prompt_tokens=3,
            num_cached_tokens=1,
        )
    )

    try:
        remaining = [partial async for partial in stream]
    finally:
        await backend.shutdown()

    assert first.completions[0].token_ids == (7,)
    assert [len(partial.completions[0].token_ids) for partial in remaining] == [4]
    final = remaining[-1]
    assert final.finished is True
    assert final.usage is not None
    assert final.usage.prompt_tokens == 3
    assert final.usage.completion_tokens == 4
    assert final.usage.cached_tokens == 1
    completion = final.completions[0]
    assert completion.text == "abcd"
    assert completion.token_ids == (7, 8, 9, 10)
    assert completion.finish_reason == "length"
    assert completion.logprobs == ({7: -0.1}, {8: -0.2}, {9: -0.3}, {10: -0.4})
    assert completion.cumulative_logprob == -1.0
    assert queue.empty()


async def test_stream_coalescing_never_discards_a_backlogged_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = KairyuBackend(num_pages=64)
    queue: asyncio.Queue[StreamUpdate] = asyncio.Queue()
    monkeypatch.setattr(backend, "_submit", lambda _request: queue)
    stream = backend.stream(_request("backlogged-error", "prompt", max_tokens=3))

    queue.put_nowait(StreamUpdate((7,), "a", False, None))
    first = await anext(stream)
    assert first.finished is False
    rich_logprob = TokenLogprob("a", 7, -0.1, (97,))
    queue.put_nowait(
        StreamUpdate(
            (7,),
            "a",
            False,
            None,
            logprobs=({7: -0.1},),
            cumulative_logprob=-0.1,
            num_prompt_tokens=3,
            num_cached_tokens=1,
            logprob_content=(rich_logprob,),
        )
    )
    failure = RuntimeError("backlogged engine failure")
    queue.put_nowait(StreamUpdate((), "", True, None, failure))

    try:
        latest = await anext(stream)
        completion = latest.completions[0]
        assert completion.text == "a"
        assert completion.logprobs == ({7: -0.1},)
        assert completion.cumulative_logprob == -0.1
        assert completion.logprob_content == (rich_logprob,)
        assert latest.usage is not None
        assert latest.usage.prompt_tokens == 3
        assert latest.usage.completion_tokens == 1
        assert latest.usage.cached_tokens == 1
        with pytest.raises(RuntimeError, match="backlogged engine failure"):
            await anext(stream)
    finally:
        await stream.aclose()
        await backend.shutdown()

    assert queue.empty()


async def test_stream_coalescing_stops_at_success_before_a_later_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = KairyuBackend(num_pages=64)
    queue: asyncio.Queue[StreamUpdate] = asyncio.Queue()
    monkeypatch.setattr(backend, "_submit", lambda _request: queue)
    stream = backend.stream(_request("terminal-boundary", "prompt", max_tokens=2))
    queue.put_nowait(StreamUpdate((7,), "d", False, None))
    queue.put_nowait(StreamUpdate((7, 8), "done", True, "length"))
    queue.put_nowait(
        StreamUpdate(
            (),
            "",
            True,
            None,
            RuntimeError("failure from a different request"),
        )
    )

    try:
        results = [partial async for partial in stream]
    finally:
        await backend.shutdown()

    assert [result.completions[0].text for result in results] == ["d", "done"]
    assert results[-1].finished is True
    assert results[-1].completions[0].finish_reason == "length"
    assert queue.qsize() == 1


def test_stream_update_queue_bounds_cumulative_backlog_and_seals_terminal() -> None:
    queue = kairyu_backend_module._StreamUpdateQueue()
    queue.publish(
        StreamUpdate(
            (),
            "",
            False,
            None,
            num_prompt_tokens=3,
            num_cached_tokens=1,
        )
    )
    for length in range(1, 128):
        queue.publish(
            StreamUpdate(
                tuple(range(length)),
                "x" * length,
                False,
                None,
                num_prompt_tokens=3,
                num_cached_tokens=1,
            )
        )
        assert queue.qsize() <= 2
    queue.publish(
        StreamUpdate(
            tuple(range(128)),
            "x" * 128,
            True,
            "length",
            num_prompt_tokens=3,
            num_cached_tokens=1,
        )
    )

    assert queue.qsize() == 2
    first = queue.get_nowait()
    final = queue.get_nowait()
    assert len(first.outputs) == 1
    assert len(final.outputs) == 128
    assert final.finished is True
    assert final.num_prompt_tokens == 3
    assert final.num_cached_tokens == 1

    queue.publish(StreamUpdate((), "", True, None, RuntimeError("late failure")))
    assert queue.empty()


def test_stream_update_queue_preserves_first_token_after_observed_empty_prefill() -> None:
    queue = kairyu_backend_module._StreamUpdateQueue()
    queue.publish(StreamUpdate((), "", False, None, num_prompt_tokens=3))
    assert queue.get_nowait().outputs == ()

    queue.publish(StreamUpdate((7,), "a", False, None, num_prompt_tokens=3))
    queue.publish(StreamUpdate((7, 8), "ab", False, None, num_prompt_tokens=3))
    queue.publish(
        StreamUpdate((7, 8, 9), "abc", True, "length", num_prompt_tokens=3)
    )

    assert queue.qsize() == 2
    assert queue.get_nowait().outputs == (7,)
    final = queue.get_nowait()
    assert final.outputs == (7, 8, 9)
    assert final.finished is True


def test_stream_update_queue_keeps_latest_snapshot_before_error() -> None:
    queue = kairyu_backend_module._StreamUpdateQueue()
    failure = RuntimeError("engine failed")
    queue.publish(StreamUpdate((7,), "a", False, None))
    for length in range(2, 65):
        queue.publish(StreamUpdate(tuple(range(length)), "x" * length, False, None))
        assert queue.qsize() <= 2
    queue.publish(StreamUpdate((), "", True, None, failure))

    assert queue.qsize() == 3
    first = queue.get_nowait()
    latest = queue.get_nowait()
    terminal = queue.get_nowait()
    assert first.outputs == (7,)
    assert len(latest.outputs) == 64
    assert terminal.error is failure
    assert queue.empty()


async def test_multi_stream_coalesces_each_choice_after_its_first_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = KairyuBackend(num_pages=64)
    queues = {
        "multi-backlog#c0": kairyu_backend_module._StreamUpdateQueue(),
        "multi-backlog#c1": kairyu_backend_module._StreamUpdateQueue(),
    }
    for index, queue in enumerate(queues.values()):
        base = 10 * (index + 1)
        queue.publish(StreamUpdate((base,), "a", False, None, num_prompt_tokens=3))
        queue.publish(
            StreamUpdate((base, base + 1), "ab", False, None, num_prompt_tokens=3)
        )
        queue.publish(
            StreamUpdate(
                (base, base + 1, base + 2),
                "abc",
                True,
                "length",
                num_prompt_tokens=3,
            )
        )

    monkeypatch.setattr(
        backend,
        "_submit",
        lambda request, **_kwargs: queues[request.request_id],
    )
    request = GenerationRequest(
        request_id="multi-backlog",
        prompt="prompt",
        sampling_params=SamplingParams(max_tokens=3, n=2),
    )

    try:
        results = [partial async for partial in backend.stream(request)]
    finally:
        await backend.shutdown()

    assert len(results) == 2
    assert [len(choice.token_ids) for choice in results[0].completions] == [1, 1]
    assert [len(choice.token_ids) for choice in results[-1].completions] == [3, 3]
    assert results[-1].finished is True
    assert results[-1].usage is not None
    assert results[-1].usage.prompt_tokens == 3
    assert results[-1].usage.completion_tokens == 6
    assert all(queue.empty() for queue in queues.values())


async def test_multi_stream_defers_choice_error_after_latest_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = KairyuBackend(num_pages=64)
    first_queue = kairyu_backend_module._StreamUpdateQueue()
    sibling_queue = kairyu_backend_module._StreamUpdateQueue()
    failure = RuntimeError("choice engine failure")
    first_queue.publish(StreamUpdate((7,), "a", False, None))
    first_queue.publish(StreamUpdate((7, 8), "ab", False, None))
    first_queue.publish(StreamUpdate((), "", True, None, failure))
    sibling_queue.publish(StreamUpdate((9,), "z", True, "length"))
    queues = {
        "multi-error#c0": first_queue,
        "multi-error#c1": sibling_queue,
    }
    monkeypatch.setattr(
        backend,
        "_submit",
        lambda request, **_kwargs: queues[request.request_id],
    )
    request = GenerationRequest(
        request_id="multi-error",
        prompt="prompt",
        sampling_params=SamplingParams(max_tokens=2, n=2),
    )
    stream = backend.stream(request)

    try:
        first = await anext(stream)
        latest = await anext(stream)
        assert first.completions[0].text == "a"
        assert latest.completions[0].text == "ab"
        with pytest.raises(RuntimeError, match="choice engine failure"):
            await anext(stream)
    finally:
        await stream.aclose()
        await backend.shutdown()

    assert all(queue.empty() for queue in queues.values())


async def test_multi_stream_yields_ready_sibling_snapshot_before_direct_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = KairyuBackend(num_pages=64)
    first_queue = kairyu_backend_module._StreamUpdateQueue()
    sibling_queue = kairyu_backend_module._StreamUpdateQueue()
    queues = {
        "multi-ready-error#c0": first_queue,
        "multi-ready-error#c1": sibling_queue,
    }
    monkeypatch.setattr(
        backend,
        "_submit",
        lambda request, **_kwargs: queues[request.request_id],
    )
    request = GenerationRequest(
        request_id="multi-ready-error",
        prompt="prompt",
        sampling_params=SamplingParams(max_tokens=3, n=2),
    )
    stream = backend.stream(request)
    first_queue.publish(StreamUpdate((1,), "a", False, None))
    sibling_queue.publish(StreamUpdate((2,), "b", False, None))

    try:
        first = await anext(stream)
        assert [choice.text for choice in first.completions] == ["a", "b"]

        failure = RuntimeError("global pump failure")
        sibling_queue.publish(StreamUpdate((2, 3), "bc", False, None))
        first_queue.publish(StreamUpdate((), "", True, None, failure))
        sibling_queue.publish(StreamUpdate((), "", True, None, failure))

        latest = await anext(stream)
        assert [choice.text for choice in latest.completions] == ["a", "bc"]
        with pytest.raises(RuntimeError, match="global pump failure"):
            await anext(stream)
    finally:
        await stream.aclose()
        await backend.shutdown()

    assert all(queue.empty() for queue in queues.values())


async def test_generation_uses_one_long_lived_step_worker(monkeypatch):
    real_to_thread = asyncio.to_thread
    invocations = []

    async def counted_to_thread(function, /, *args, **kwargs):
        invocations.append(function)
        return await real_to_thread(function, *args, **kwargs)

    monkeypatch.setattr(kairyu_backend_module.asyncio, "to_thread", counted_to_thread)
    backend = KairyuBackend(num_pages=256)

    result = await backend.generate(_request("one-worker", "prompt", max_tokens=64))

    assert len(result.completions[0].token_ids) == 64
    worker_invocations = [
        function
        for function in invocations
        if (
            getattr(function, "__self__", None) is backend
            and getattr(function, "__func__", None) is KairyuBackend._step_worker
        )
    ]
    assert len(worker_invocations) == 1
    await backend.shutdown()


async def test_long_lived_worker_publishes_each_step_before_consumer_coalesces():
    runner = _StepGateRunner(gated_call=2)
    backend = KairyuBackend(num_pages=256, runner=runner)
    stream = backend.stream(_request("cadence", "prompt", max_tokens=3))
    partials = []
    try:
        first = await asyncio.wait_for(anext(stream), timeout=1)
        partials.append(first)
        for _ in range(100):
            if runner.entered.is_set():
                break
            await asyncio.sleep(0.01)

        assert runner.entered.is_set()
        assert first.finished is False
        assert len(first.completions[0].token_ids) == 1
        assert backend._pump_task is not None
        assert not backend._pump_task.done()

        runner.allow_step.set()
        queue = backend._queues["cadence"]
        for _ in range(100):
            if queue.qsize() == 1:
                break
            await asyncio.sleep(0.01)
        assert queue.qsize() == 1
        async for partial in stream:
            partials.append(partial)
    finally:
        runner.allow_step.set()
        await stream.aclose()
        await backend.shutdown()

    assert [len(partial.completions[0].token_ids) for partial in partials] == [
        1,
        3,
    ]
    assert partials[-1].finished is True
    assert runner.calls == 3


async def test_submit_after_worker_idle_snapshot_restarts_without_stranding():
    backend = KairyuBackend(num_pages=256)
    original_has_work = backend._loop.has_work
    worker_saw_idle = threading.Event()
    release_idle_worker = threading.Event()
    event_loop_thread = threading.get_ident()

    def pause_idle_worker() -> bool:
        has_work = original_has_work()
        if (
            threading.get_ident() != event_loop_thread
            and not has_work
            and not worker_saw_idle.is_set()
        ):
            worker_saw_idle.set()
            if not release_idle_worker.wait(timeout=2):
                raise TimeoutError("test did not release idle step worker")
        return has_work

    backend._loop.has_work = pause_idle_worker
    try:
        first = await backend.generate(_request("before-idle", "first", max_tokens=1))
        assert first.finished
        for _ in range(100):
            if worker_saw_idle.is_set():
                break
            await asyncio.sleep(0.01)
        assert worker_saw_idle.is_set()

        second_task = asyncio.create_task(
            backend.generate(_request("after-idle", "second", max_tokens=2))
        )
        await asyncio.sleep(0)
        assert not second_task.done()
        release_idle_worker.set()

        second = await asyncio.wait_for(second_task, timeout=2)
        assert second.finished
    finally:
        release_idle_worker.set()
        await backend.shutdown()


async def test_submit_while_worker_steps_does_not_read_scheduler_from_event_loop():
    runner = _StepGateRunner(gated_call=1)
    backend = KairyuBackend(num_pages=256, runner=runner)
    original_has_work = backend._loop.has_work
    event_loop_thread = threading.get_ident()
    concurrent_event_loop_reads = 0

    def record_has_work() -> bool:
        nonlocal concurrent_event_loop_reads
        if (
            threading.get_ident() == event_loop_thread
            and runner.entered.is_set()
            and not runner.allow_step.is_set()
        ):
            concurrent_event_loop_reads += 1
        return original_has_work()

    backend._loop.has_work = record_has_work
    first = asyncio.create_task(
        backend.generate(_request("active-worker", "first", max_tokens=2))
    )
    try:
        assert await asyncio.to_thread(runner.entered.wait, 2)
        second = asyncio.create_task(
            backend.generate(_request("submitted-during-step", "second", max_tokens=2))
        )
        await asyncio.sleep(0)

        assert not second.done()
        assert concurrent_event_loop_reads == 0

        runner.allow_step.set()
        first_result, second_result = await asyncio.wait_for(
            asyncio.gather(first, second),
            timeout=2,
        )
        assert first_result.finished
        assert second_result.finished
    finally:
        runner.allow_step.set()
        if not first.done():
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)
        await backend.shutdown()


async def test_stream_abandonment_aborts_engine_work():
    backend = KairyuBackend(num_pages=256, runner=_SlowRunner())
    stream = backend.stream(_request("abandoned", "keep generating", max_tokens=10_000))
    first = await anext(stream)
    assert first.finished is False

    await stream.aclose()
    for _ in range(100):
        if not backend._loop.has_work():
            break
        await asyncio.sleep(0.01)

    assert backend._loop.has_work() is False
    assert "abandoned" not in backend._scheduler.states
    result = await backend.generate(_request("after-abandon", "still works", 2))
    assert result.finished


async def test_multi_completion_abandonment_aborts_all_siblings():
    backend = KairyuBackend(num_pages=256, runner=_SlowRunner())
    request = GenerationRequest(
        request_id="multi-abandon",
        prompt="keep generating siblings",
        sampling_params=SamplingParams(max_tokens=10_000, n=3),
    )
    stream = backend.stream(request)
    await anext(stream)
    pump = backend._pump_task
    assert pump is not None
    pump.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pump
    assert backend._pump_task is None

    await stream.aclose()
    for _ in range(100):
        if not backend._loop.has_work():
            break
        await asyncio.sleep(0.01)
    assert backend._loop.has_work() is False
    assert not {
        "multi-abandon#c0",
        "multi-abandon#c1",
        "multi-abandon#c2",
    } & set(backend._scheduler.states)


async def test_pump_failure_purges_requests_and_backend_recovers(monkeypatch):
    real_to_thread = asyncio.to_thread
    worker_invocations = 0

    async def counted_to_thread(function, /, *args, **kwargs):
        nonlocal worker_invocations
        if (
            getattr(function, "__func__", None) is KairyuBackend._step_worker
        ):
            worker_invocations += 1
        return await real_to_thread(function, *args, **kwargs)

    monkeypatch.setattr(kairyu_backend_module.asyncio, "to_thread", counted_to_thread)
    backend = KairyuBackend(num_pages=256, runner=_FailOnceRunner())
    with pytest.raises(RuntimeError, match="injected runner failure"):
        await backend.generate(_request("failed", "first", 2))

    assert backend._loop.has_work() is False
    assert "failed" not in backend._scheduler.states
    recovered = await backend.generate(_request("recovered", "second", 2))
    assert recovered.finished
    assert worker_invocations == 2
    await backend.shutdown()


async def test_shutdown_waits_for_step_boundary_and_stops_worker():
    runner = _StepGateRunner(gated_call=1)
    backend = KairyuBackend(num_pages=256, runner=runner)
    generation = asyncio.create_task(
        backend.generate(_request("shutdown", "prompt", max_tokens=10_000))
    )
    for _ in range(100):
        if runner.entered.is_set():
            break
        await asyncio.sleep(0.01)
    assert runner.entered.is_set()

    shutdown = asyncio.create_task(backend.shutdown())
    await asyncio.sleep(0.01)
    assert not shutdown.done()

    runner.allow_step.set()
    await asyncio.wait_for(shutdown, timeout=2)

    assert runner.calls == 1
    assert backend._pump_task is None
    assert backend._loop.has_work() is False
    generation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await generation
    with pytest.raises(RuntimeError, match="shut down"):
        await backend.generate(_request("after-shutdown", "prompt", max_tokens=1))


async def test_shutdown_is_shared_and_cancellation_safe_through_all_cleanup():
    backend = KairyuBackend(num_pages=256)
    close_started = threading.Event()
    allow_close = threading.Event()
    launcher_started = threading.Event()
    allow_launcher = threading.Event()
    real_close = backend._loop.close
    close_calls = 0

    def blocking_close() -> None:
        nonlocal close_calls
        close_calls += 1
        close_started.set()
        if not allow_close.wait(timeout=2):
            raise TimeoutError("test did not release engine-loop close")
        real_close()

    class _BlockingLauncher:
        def __init__(self) -> None:
            self.shutdown_calls = 0

        def shutdown(self) -> None:
            self.shutdown_calls += 1
            launcher_started.set()
            if not allow_launcher.wait(timeout=2):
                raise TimeoutError("test did not release TP launcher shutdown")

    launcher = _BlockingLauncher()
    backend._loop.close = blocking_close
    backend._loop.parallel_launcher = launcher

    first = asyncio.create_task(backend.shutdown())
    try:
        assert await asyncio.to_thread(close_started.wait, 2)
        second = asyncio.create_task(backend.shutdown())
        await asyncio.sleep(0)
        assert not second.done()

        first.cancel()
        await asyncio.sleep(0)
        assert not first.done()
        assert not second.done()

        allow_close.set()
        assert await asyncio.to_thread(launcher_started.wait, 2)
        first.cancel()
        await asyncio.sleep(0)
        assert not first.done()
        assert not second.done()

        allow_launcher.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(first, timeout=2)
        await asyncio.wait_for(second, timeout=2)
    finally:
        allow_close.set()
        allow_launcher.set()

    assert close_calls == 1
    assert launcher.shutdown_calls == 1
    await backend.shutdown()
    assert close_calls == 1
    assert launcher.shutdown_calls == 1


async def test_request_submitted_during_failure_purge_is_pumped():
    backend = KairyuBackend(num_pages=256, runner=_FailOnceRunner())
    purge_started = threading.Event()
    allow_purge = threading.Event()
    original_purge = backend._loop.purge
    original_has_work = backend._loop.has_work
    event_loop_thread = threading.get_ident()
    concurrent_event_loop_reads = 0

    def delayed_purge(request_ids: tuple[str, ...]) -> None:
        purge_started.set()
        if not allow_purge.wait(timeout=2):
            raise TimeoutError("test did not release purge")
        original_purge(request_ids)

    def record_has_work() -> bool:
        nonlocal concurrent_event_loop_reads
        if (
            threading.get_ident() == event_loop_thread
            and purge_started.is_set()
            and not allow_purge.is_set()
        ):
            concurrent_event_loop_reads += 1
        return original_has_work()

    backend._loop.purge = delayed_purge
    backend._loop.has_work = record_has_work
    failed = asyncio.create_task(backend.generate(_request("failed", "first", 2)))
    assert await asyncio.to_thread(purge_started.wait, 2)

    recovered = asyncio.create_task(
        backend.generate(_request("submitted-during-purge", "second", 2))
    )
    await asyncio.sleep(0)
    assert concurrent_event_loop_reads == 0
    allow_purge.set()

    with pytest.raises(RuntimeError, match="injected runner failure"):
        await failed
    result = await asyncio.wait_for(recovered, timeout=2)
    assert result.finished


async def test_duplicate_request_id_is_rejected_without_disrupting_active_request():
    backend = KairyuBackend(num_pages=256, runner=_SlowRunner())
    first = asyncio.create_task(backend.generate(_request("duplicate", "first", 10_000)))
    for _ in range(100):
        if "duplicate" in backend._loop._active_request_ids:
            break
        await asyncio.sleep(0.01)
    assert "duplicate" in backend._loop._active_request_ids

    with pytest.raises(ValueError, match="duplicate request_id"):
        await backend.generate(_request("duplicate", "second"))

    assert not first.done()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    for _ in range(100):
        if "duplicate" not in backend._loop._active_request_ids:
            break
        await asyncio.sleep(0.01)
    assert "duplicate" not in backend._loop._active_request_ids

    reused = await backend.generate(_request("duplicate", "reused", 2))
    assert reused.finished is True


def test_unknown_abort_does_not_reserve_request_id():
    backend = KairyuBackend(num_pages=256)
    backend._loop.abort("unknown")
    backend._loop.submit("unknown", "prompt", SamplingParams(max_tokens=1))
    backend._loop.purge(("unknown",))


@pytest.mark.parametrize("duplicate_api", ["generate", "stream"])
async def test_duplicate_rejected_after_worker_forget_before_terminal_delivery(
    duplicate_api,
):
    backend = KairyuBackend(num_pages=256, runner=_SlowRunner())
    forgot_request = threading.Event()
    allow_step_return = threading.Event()
    original_forget = backend._loop._forget

    def forget_then_block(request_id: str) -> None:
        original_forget(request_id)
        forgot_request.set()
        if not allow_step_return.wait(timeout=2):
            raise TimeoutError("test did not release terminal step")

    backend._loop._forget = forget_then_block
    first = asyncio.create_task(backend.generate(_request("delivery-gap", "first", 1)))
    assert await asyncio.to_thread(forgot_request.wait, 2)
    original_queue = backend._queues["delivery-gap"]
    assertions_complete = False

    async def submit_duplicate() -> None:
        duplicate = _request("delivery-gap", "duplicate", 1)
        if duplicate_api == "generate":
            await backend.generate(duplicate)
        else:
            await anext(backend.stream(duplicate))

    try:
        with pytest.raises(ValueError, match="duplicate request_id"):
            await asyncio.wait_for(submit_duplicate(), timeout=0.2)
        assert backend._queues["delivery-gap"] is original_queue
        assertions_complete = True
    finally:
        allow_step_return.set()
        if not assertions_complete:
            first.cancel()
            try:
                await first
            except asyncio.CancelledError:
                pass

    result = await asyncio.wait_for(first, timeout=2)
    assert result.finished
    reused = await backend.generate(_request("delivery-gap", "reused", 1))
    assert reused.finished


async def test_duplicate_rejected_after_fatal_purge_before_failure_delivery():
    backend = KairyuBackend(num_pages=256, runner=_FailOnceRunner())
    purged_request = threading.Event()
    allow_purge_return = threading.Event()
    original_purge = backend._loop.purge

    def purge_then_block(request_ids: tuple[str, ...]) -> None:
        original_purge(request_ids)
        purged_request.set()
        if not allow_purge_return.wait(timeout=2):
            raise TimeoutError("test did not release fatal purge")

    backend._loop.purge = purge_then_block
    first = asyncio.create_task(backend.generate(_request("purge-gap", "first", 1)))
    assert await asyncio.to_thread(purged_request.wait, 2)
    original_queue = backend._queues["purge-gap"]
    assertions_complete = False

    try:
        with pytest.raises(ValueError, match="duplicate request_id"):
            await asyncio.wait_for(
                backend.generate(_request("purge-gap", "duplicate", 1)), timeout=0.2
            )
        assert backend._queues["purge-gap"] is original_queue
        assertions_complete = True
    finally:
        allow_purge_return.set()
        if not assertions_complete:
            first.cancel()
            try:
                await first
            except asyncio.CancelledError:
                pass

    with pytest.raises(RuntimeError, match="injected runner failure"):
        await asyncio.wait_for(first, timeout=2)
    reused = await backend.generate(_request("purge-gap", "reused", 1))
    assert reused.finished


async def test_multi_stream_reserves_all_sub_ids_before_submit():
    backend = KairyuBackend(num_pages=256, runner=_SlowRunner())
    blocker = asyncio.create_task(backend.generate(_request("atomic#c1", "blocker", 10_000)))
    for _ in range(100):
        if "atomic#c1" in backend._loop._active_request_ids:
            break
        await asyncio.sleep(0.01)
    assert "atomic#c1" in backend._loop._active_request_ids
    blocker_queue = backend._queues["atomic#c1"]
    request = GenerationRequest(
        request_id="atomic",
        prompt="multi",
        sampling_params=SamplingParams(max_tokens=10_000, n=3),
    )

    try:
        with pytest.raises(ValueError, match="duplicate request_id"):
            await anext(backend.stream(request))
        assert backend._queues["atomic#c1"] is blocker_queue
        assert "atomic#c0" not in backend._queues
        assert "atomic#c2" not in backend._queues
        assert not {"atomic#c0", "atomic#c2"} & backend._active_request_ids
    finally:
        for leaked_id in ("atomic#c0", "atomic#c2"):
            backend._abort(leaked_id)
            backend._queues.pop(leaked_id, None)
        blocker.cancel()
        try:
            await blocker
        except asyncio.CancelledError:
            pass


def test_multi_candidate_sub_requests_preserve_parallel_tool_intent():
    backend = KairyuBackend(num_pages=64)
    request = GenerationRequest(
        request_id="multi-tools",
        prompt="prompt",
        sampling_params=SamplingParams(max_tokens=1, n=2),
        parallel_tool_calls=False,
    )

    sub_requests = backend._sub_requests(request)

    assert [sub.parallel_tool_calls for sub in sub_requests] == [False, False]


async def test_multi_stream_partial_submit_failure_rolls_back_all_sub_ids():
    backend = KairyuBackend(num_pages=256)
    real_submit = backend._loop.submit
    submit_count = 0

    def fail_second_submit(
        request_id,
        prompt,
        params,
        priority=0,
        scheduling_class="interactive",
        prepared_prompt=None,
        trace_requested=False,
    ):
        nonlocal submit_count
        submit_count += 1
        if submit_count == 2:
            raise RuntimeError("injected submit failure")
        real_submit(
            request_id,
            prompt,
            params,
            priority=priority,
            scheduling_class=scheduling_class,
            prepared_prompt=prepared_prompt,
            trace_requested=trace_requested,
        )

    backend._loop.submit = fail_second_submit
    request = GenerationRequest(
        request_id="partial",
        prompt="multi",
        sampling_params=SamplingParams(max_tokens=10_000, n=3),
    )
    sub_ids = {"partial#c0", "partial#c1", "partial#c2"}

    try:
        with pytest.raises(RuntimeError, match="injected submit failure"):
            await anext(backend.stream(request))
        assert not sub_ids & set(backend._queues)
    finally:
        backend._loop.submit = real_submit
        for leaked_id in sub_ids:
            backend._abort(leaked_id)
            backend._queues.pop(leaked_id, None)
        for _ in range(100):
            if not sub_ids & backend._loop._active_request_ids:
                break
            await asyncio.sleep(0.01)
        assert not sub_ids & backend._loop._active_request_ids
        assert not sub_ids & backend._active_request_ids


async def test_single_submit_failure_rolls_back_public_id_reservation():
    backend = KairyuBackend(num_pages=256)
    real_submit = backend._loop.submit

    def fail_submit(
        request_id,
        prompt,
        params,
        priority=0,
        scheduling_class="interactive",
        prepared_prompt=None,
        trace_requested=False,
    ):
        raise RuntimeError("injected submit failure")

    backend._loop.submit = fail_submit
    with pytest.raises(RuntimeError, match="injected submit failure"):
        await backend.generate(_request("submit-failure", "first", 1))
    assert "submit-failure" not in backend._active_request_ids
    assert "submit-failure" not in backend._queues

    backend._loop.submit = real_submit
    reused = await backend.generate(_request("submit-failure", "reused", 1))
    assert reused.finished


async def test_multi_generate_reserves_each_sub_for_aggregate_lifetime():
    backend = KairyuBackend(num_pages=256, runner=_SlowRunner())
    request = GenerationRequest(
        request_id="multi-generate",
        prompt="multi",
        sampling_params=SamplingParams(max_tokens=10_000, n=3),
    )
    sub_ids = {"multi-generate#c0", "multi-generate#c1", "multi-generate#c2"}
    first = asyncio.create_task(backend.generate(request))
    for _ in range(100):
        if sub_ids <= backend._active_request_ids:
            break
        await asyncio.sleep(0.01)
    assert sub_ids <= backend._active_request_ids
    original_queues = {request_id: backend._queues[request_id] for request_id in sub_ids}

    with pytest.raises(ValueError, match="duplicate request_id"):
        await backend.generate(request)
    assert all(
        backend._queues[request_id] is queue for request_id, queue in original_queues.items()
    )

    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert not sub_ids & backend._active_request_ids
    for _ in range(100):
        if not sub_ids & backend._loop._active_request_ids:
            break
        await asyncio.sleep(0.01)
    assert not sub_ids & backend._loop._active_request_ids

    reused = await backend.generate(
        GenerationRequest(
            request_id="multi-generate",
            prompt="reused",
            sampling_params=SamplingParams(max_tokens=1, n=3),
        )
    )
    assert reused.finished


async def test_concurrent_requests_are_continuously_batched():
    backend = KairyuBackend(num_pages=256)
    results = await asyncio.gather(
        backend.generate(_request("a", "first prompt")),
        backend.generate(_request("b", "second prompt")),
        backend.generate(_request("c", "third prompt")),
    )
    assert all(r.finished for r in results)
    assert {r.request_id for r in results} == {"a", "b", "c"}


def test_registered_as_kairyu_backend():
    backend = create_backend("kairyu", num_pages=64)
    assert isinstance(backend, KairyuBackend)


def test_tensor_parallel_size_recorded():
    backend = KairyuBackend(num_pages=64, tensor_parallel_size=2)
    assert backend.tensor_parallel_size == 2


def test_expert_parallel_size_and_topology_metadata_are_recorded(monkeypatch):
    loop, cache, scheduler = build_engine_loop(num_pages=64)
    loop.parallelism_metadata = {
        "parallelism": "expert_parallel",
        "expert_parallel_size": 4,
        "attention_placement": "replicated",
        "attention_output_placement": "row_parallel",
        "attention_output_parallel_size": 4,
        "attention_output_partial_dtype": "bfloat16",
        "execution_mode": "replicated-attention-correctness",
        "pipeline_depth": 1,
        "decode_mode": "eager",
        "kv_cache_dtype": "bfloat16",
    }
    captured = None

    def fake_build_engine_loop(**kwargs):
        nonlocal captured
        captured = kwargs
        return loop, cache, scheduler

    monkeypatch.setattr(
        kairyu_backend_module,
        "build_engine_loop",
        fake_build_engine_loop,
    )

    backend = KairyuBackend(
        expert_parallel_size=4,
        expert_parallel_attention_dp=True,
    )

    assert captured is not None
    assert captured["expert_parallel_size"] == 4
    assert captured["expert_parallel_attention_dp"] is True
    assert backend.expert_parallel_size == 4
    assert backend.expert_parallel_attention_dp is True
    assert backend.parallelism_metadata == loop.parallelism_metadata
    loop.close()


def test_custom_runner_requires_one_distinct_runner_per_tp_rank():
    with pytest.raises(
        ValueError,
        match=(
            r"custom runner.*one distinct rank-local runner per rank.*"
            r"model_path.*tensor_parallel_size=1"
        ),
    ):
        build_engine_loop(runner=object(), tensor_parallel_size=2)


def test_custom_runner_remains_supported_at_tp1():
    custom = object()
    loop, _cache, _scheduler = build_engine_loop(runner=custom)
    assert loop._runner is custom


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"expert_parallel_size": 3}, "expert_parallel_size"),
        (
            {
                "model_path": "/unused",
                "tensor_parallel_size": 0,
                "expert_parallel_size": 2,
                "kv_cache_dtype": "bfloat16",
            },
            "tensor_parallel_size",
        ),
        (
            {
                "model_path": "/unused",
                "tensor_parallel_size": True,
                "expert_parallel_size": 2,
                "kv_cache_dtype": "bfloat16",
            },
            "tensor_parallel_size",
        ),
        (
            {
                "model_path": "/unused",
                "tensor_parallel_size": 2,
                "expert_parallel_size": 2,
                "kv_cache_dtype": "bfloat16",
            },
            "mutually exclusive",
        ),
        ({"expert_parallel_size": 2}, "real model_path"),
        (
            {"model_path": "/unused", "expert_parallel_size": 2},
            "kv_cache_dtype='bfloat16'",
        ),
        (
            {
                "model_path": "/unused",
                "expert_parallel_size": 2,
                "kv_cache_dtype": "bfloat16",
                "pipeline_depth": 2,
            },
            "pipeline_depth=1",
        ),
        (
            {
                "model_path": "/unused",
                "expert_parallel_size": 2,
                "kv_cache_dtype": "bfloat16",
                "speculative": "ngram",
            },
            "speculative decoding",
        ),
        (
            {
                "model_path": "/unused",
                "expert_parallel_size": 2,
                "kv_cache_dtype": "bfloat16",
                "decode_mode": "cuda_graph",
            },
            "decode_mode='eager'",
        ),
    ],
)
def test_build_engine_loop_rejects_unsupported_expert_parallel_options(
    options,
    message,
):
    with pytest.raises(ValueError, match=message):
        build_engine_loop(**options)


@pytest.mark.parametrize("degree", [0, 3])
def test_tensor_parallel_size_rejects_invalid_degrees(degree):
    with pytest.raises(ValueError):
        KairyuBackend(num_pages=64, tensor_parallel_size=degree)


async def test_tp2_greedy_output_equals_tp1():
    # Deterministic toy ranks + FakeCommunicator: TP must not change sampling
    # (design m5 D1: same step, identical samples on every rank).
    prompts = ["hello world from kairyu", "another prompt entirely"]
    results = {}
    for degree in (1, 2):
        backend = KairyuBackend(num_pages=256, tensor_parallel_size=degree)
        outputs = await asyncio.gather(
            *(
                backend.generate(_request(f"tp{degree}-{i}", prompt, max_tokens=5))
                for i, prompt in enumerate(prompts)
            )
        )
        results[degree] = [result.completions[0].token_ids for result in outputs]
    assert results[1] == results[2]


def test_registry_forwards_tensor_parallel_size():
    backend = create_backend("kairyu", num_pages=64, tensor_parallel_size=4)
    assert isinstance(backend, KairyuBackend)
    assert backend.tensor_parallel_size == 4


def test_registry_forwards_server_max_model_len_option():
    backend = create_backend("kairyu", num_pages=64, max_model_len=8192)

    assert isinstance(backend, KairyuBackend)
    assert backend._loop.max_model_len == 8192
    backend._loop.close()


@pytest.mark.parametrize(
    ("builder_name", "options"),
    [
        ("_build_dist_tp_loop", {"model_path": "/unused", "tensor_parallel_size": 2}),
        (
            "_build_dist_ep_loop",
            {
                "model_path": "/unused",
                "expert_parallel_size": 4,
                "kv_cache_dtype": "bfloat16",
            },
        ),
        ("_build_pd_loop", {"model_path": "/unused", "pd_separation": True}),
    ],
)
def test_context_limit_crosses_distributed_and_pd_builders(
    monkeypatch,
    builder_name,
    options,
):
    captured = None
    sentinel = (object(), object(), object())

    def fake_builder(**kwargs):
        nonlocal captured
        captured = kwargs
        return sentinel

    monkeypatch.setattr(kairyu_backend_module, builder_name, fake_builder)

    result = build_engine_loop(max_model_len=8192, **options)

    assert result is sentinel
    assert captured is not None
    assert captured["max_model_len"] == 8192


def test_dist_ep_builder_assembles_topology_neutral_serving_loop(
    monkeypatch,
    tmp_path,
):
    (tmp_path / "config.json").write_text(
        '{"vocab_size": 50000}',
        encoding="utf-8",
    )
    created = []

    class _FakeEPLauncher:
        def __init__(
            self,
            model_path,
            expert_parallel_size,
            num_pages,
            page_size,
            **kwargs,
        ):
            self.model_path = model_path
            self.expert_parallel_size = expert_parallel_size
            self.num_pages = num_pages
            self.page_size = page_size
            self.kwargs = kwargs
            self.runner = object()
            self.attention_backend_decision = None
            self.kv_cache_dtype_requested = "bfloat16"
            self.kv_cache_dtype_resolved = "bfloat16"
            self.shutdown_calls = 0
            created.append(self)

        def parallelism_metadata(self):
            return {
                "parallelism": "expert_parallel",
                "expert_parallel_size": self.expert_parallel_size,
                "attention_placement": "replicated",
                "attention_output_placement": "row_parallel",
                "attention_output_parallel_size": self.expert_parallel_size,
                "attention_output_partial_dtype": "bfloat16",
                "execution_mode": "replicated-attention-correctness",
                "pipeline_depth": 1,
                "decode_mode": "eager",
                "kv_cache_dtype": "bfloat16",
            }

        def shutdown(self):
            self.shutdown_calls += 1

    monkeypatch.setattr(
        "kairyu.engine.core.worker.DistEPLauncher",
        _FakeEPLauncher,
    )

    loop, _cache, _scheduler = kairyu_backend_module._build_dist_ep_loop(
        model_path=str(tmp_path),
        expert_parallel_size=4,
        num_pages=64,
        page_size=16,
        max_num_batched_tokens=32,
        max_num_seqs=4,
        max_model_len=8192,
        priority_age_s=60.0,
        tokenizer=ToyTokenizer(),
        pipeline_depth=1,
        decode_mode="eager",
        kv_cache_dtype="bfloat16",
    )

    try:
        launcher = created[0]
        assert loop._runner is launcher.runner
        assert loop.parallel_launcher is launcher
        assert loop.ep_launcher is launcher
        assert loop.tp_launcher is None
        assert loop.parallelism_metadata == launcher.parallelism_metadata()
        assert launcher.kwargs["pipeline_depth"] == 1
        assert launcher.kwargs["decode_mode"] == "eager"
        assert launcher.kwargs["kv_cache_dtype"] == "bfloat16"
    finally:
        loop.close()


def test_dist_ep_attention_dp_propagates_cuda_graph_capacity_and_scratch(
    monkeypatch,
    tmp_path,
):
    (tmp_path / "config.json").write_text('{"vocab_size": 50000}')
    created = []

    class _FakeEPLauncher:
        def __init__(
            self,
            model_path,
            expert_parallel_size,
            num_pages,
            page_size,
            **kwargs,
        ):
            self.runner = object()
            self.kwargs = kwargs
            self.attention_backend_decision = None
            self.kv_cache_dtype_requested = "bfloat16"
            self.kv_cache_dtype_resolved = "bfloat16"
            created.append(self)

        def parallelism_metadata(self):
            return {
                "parallelism": "expert_parallel",
                "expert_parallel_size": 4,
                "attention_placement": "request_owned_data_parallel",
                "attention_output_placement": "replicated",
                "attention_output_parallel_size": 1,
                "attention_output_partial_dtype": None,
                "execution_mode": "request-owned-attention-dp",
                "pipeline_depth": 5,
                "decode_mode": "cuda_graph",
                "kv_cache_dtype": "bfloat16",
            }

        def shutdown(self):
            pass

    monkeypatch.setattr(
        "kairyu.engine.core.worker.DistEPLauncher",
        _FakeEPLauncher,
    )
    loop, _cache, _scheduler = kairyu_backend_module._build_dist_ep_loop(
        model_path=str(tmp_path),
        expert_parallel_size=4,
        expert_parallel_attention_dp=True,
        num_pages=64,
        page_size=16,
        max_num_batched_tokens=64,
        max_num_seqs=16,
        max_model_len=2048,
        priority_age_s=60.0,
        tokenizer=ToyTokenizer(),
        pipeline_depth=5,
        decode_mode="cuda_graph",
        cuda_graph_max_batch=8,
        cuda_graph_max_pages=32,
        cuda_graph_warmup_iters=2,
        kv_cache_dtype="bfloat16",
    )

    try:
        assert created[0].kwargs["graph_scratch_page"] == 74
        assert created[0].kwargs["graph_max_batch"] == 8
        assert created[0].kwargs["graph_max_pages"] == 32
        assert created[0].kwargs["graph_warmup_iters"] == 2
        assert created[0].kwargs["decode_mode"] == "cuda_graph"
    finally:
        loop.close()


async def test_full_stack_openai_server_over_engine_core():
    app = create_app(
        engines={"kairyu-cpu": KairyuBackend(num_pages=256)},
        legacy_chat_models={"kairyu-cpu"},
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "kairyu-cpu",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4,
            },
        )
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"]
    assert data["choices"][0]["finish_reason"] == "length"


async def test_openai_server_enforces_configured_context_limit():
    backend = KairyuBackend(num_pages=64, max_model_len=5)
    app = create_app(engines={"limited": backend})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        oversized = await client.post(
            "/v1/completions",
            json={
                "model": "limited",
                "prompt": [1, 2, 3, 4],
                "max_tokens": 2,
            },
        )
        boundary = await client.post(
            "/v1/completions",
            json={
                "model": "limited",
                "prompt": [1, 2, 3],
                "max_tokens": 2,
            },
        )

    assert oversized.status_code == 400
    assert "exceed max_model_len" in oversized.json()["error"]["message"]
    assert boundary.status_code == 200
    usage = boundary.json()["usage"]
    assert usage["prompt_tokens"] == 3
    assert usage["completion_tokens"] == 2
    assert usage["total_tokens"] == 5
    await backend.shutdown()


class _AlwaysFailRunner:
    def execute(
        self,
        scheduled: tuple[ScheduledChunk, ...],
        states: Mapping[str, object],
    ) -> dict[str, tuple[SampledToken, ...]]:
        raise RuntimeError("transport is gone")


class _DeadRankLauncher:
    def __init__(self, dead: tuple[int, ...], failure: str | None = None) -> None:
        self._dead = dead
        self._failure = failure

    def dead_ranks(self) -> tuple[int, ...]:
        return self._dead

    def failure_type(self) -> str | None:
        return self._failure


async def test_readiness_is_ready_before_any_work():
    backend = KairyuBackend(runner=_SlowRunner())
    assert backend.readiness().ready is True


async def test_a_step_failure_reports_but_does_not_flip_readiness():
    # review [P1] on #126: marking the node unready here is a trap. The load
    # balancer stops sending work, so the successful step that would clear the
    # flag never arrives and the node stays out of rotation until someone looks.
    # A step error also cannot be told apart from one bad request.
    backend = KairyuBackend(runner=_AlwaysFailRunner())
    with pytest.raises(RuntimeError):
        await backend.generate(_request("r1", "hello"))
    status = backend.readiness()
    assert status.ready is True
    assert status.fatal is False
    assert "RuntimeError" in status.detail  # reported for diagnosis


async def test_the_reported_error_detail_carries_no_message():
    # /readyz is unauthenticated; an exception message can carry a path or URL
    class _LeakyRunner:
        def execute(self, scheduled, states):
            raise RuntimeError("connect https://internal:8443 failed for /etc/key")

    backend = KairyuBackend(runner=_LeakyRunner())
    with pytest.raises(RuntimeError):
        await backend.generate(_request("r1", "hello"))
    detail = backend.readiness().detail
    assert "RuntimeError" in detail
    assert "internal" not in detail
    assert "/etc/key" not in detail


async def test_readiness_reports_dead_tensor_parallel_ranks_as_fatal():
    # nothing in-process can bring a dead rank back, so this is the one condition
    # that both stops traffic AND asks for a restart
    backend = KairyuBackend(runner=_SlowRunner())
    backend._loop.parallel_launcher = _DeadRankLauncher((2, 5))
    status = backend.readiness()
    assert status.ready is False
    assert status.fatal is True
    assert status.detail.startswith("tensor-parallel ranks not running:")
    assert "[2, 5]" in status.detail


async def test_live_tensor_parallel_ranks_stay_ready():
    backend = KairyuBackend(runner=_SlowRunner())
    backend._loop.parallel_launcher = _DeadRankLauncher(())
    assert backend.readiness().ready is True


async def test_failed_tensor_parallel_transport_is_fatal_even_with_live_ranks():
    backend = KairyuBackend(runner=_SlowRunner())
    backend._loop.parallel_launcher = _DeadRankLauncher(
        (),
        failure="DistBackendError",
    )
    status = backend.readiness()
    assert status.ready is False
    assert status.fatal is True
    assert status.detail == "tensor-parallel transport failed: DistBackendError"


async def test_readiness_names_expert_parallel_failures_without_calling_them_tp():
    backend = KairyuBackend(runner=_SlowRunner())
    backend._loop.parallel_launcher = _DeadRankLauncher((3,))
    backend._loop.parallelism_metadata = {
        "parallelism": "expert_parallel",
    }

    status = backend.readiness()

    assert status.ready is False
    assert status.fatal is True
    assert status.detail == "expert-parallel ranks not running: [3]"
