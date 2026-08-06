"""Contract tests for vLLM AsyncLLMEngine-style usage (design doc D2).

Mirrors the vLLM pattern:
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(model=...))
    async for output in engine.generate(prompt, params, request_id): ...
"""

import asyncio

import pytest

from kairyu import (
    AsyncEngineArgs,
    AsyncLLMEngine,
    RequestOutput,
    SamplingParams,
    TokensPrompt,
)
from kairyu.engine.mock import MockBackend


def _engine() -> AsyncLLMEngine:
    return AsyncLLMEngine(backend=MockBackend(), model="mock-model")


async def test_generate_yields_incremental_then_finished_request_outputs():
    engine = _engine()
    outputs = []
    async for output in engine.generate(
        "a reasonably long prompt to stream", SamplingParams(max_tokens=32), "req-1"
    ):
        assert isinstance(output, RequestOutput)
        assert output.request_id == "req-1"
        outputs.append(output)
    assert len(outputs) > 1
    assert outputs[-1].finished is True
    assert all(o.finished is False for o in outputs[:-1])
    final_text = outputs[-1].outputs[0].text
    assert outputs[0].outputs[0].text == final_text[: len(outputs[0].outputs[0].text)]


async def test_from_engine_args_constructs_engine():
    args = AsyncEngineArgs(model="mock-model", enable_prefix_caching=True)
    engine = AsyncLLMEngine.from_engine_args(args)
    async for output in engine.generate("hi", SamplingParams(), "req-2"):
        last = output
    assert last.finished is True
    assert last.outputs[0].text


async def test_token_prompt_preserves_exact_input_ids_in_request_output():
    engine = _engine()

    async for output in engine.generate(
        TokensPrompt((2, 7, 1)),
        SamplingParams(max_tokens=2),
        "token-request",
    ):
        last = output

    assert last.prompt is None
    assert last.prompt_token_ids == (2, 7, 1)


async def test_token_prompt_is_not_streamed_to_an_undeclared_legacy_backend():
    class LegacyBackend:
        calls = 0

        async def stream(self, _request):
            self.calls += 1
            raise AssertionError("typed prompt was dispatched")
            yield  # pragma: no cover

    backend = LegacyBackend()
    engine = AsyncLLMEngine(backend=backend, model="legacy")

    with pytest.raises(ValueError, match="text-only"):
        async for _ in engine.generate(
            TokensPrompt((1, 2, 3)),
            SamplingParams(max_tokens=1),
            "legacy-token",
        ):
            pass

    assert backend.calls == 0


async def test_abort_is_accepted():
    engine = _engine()
    await engine.abort("future")  # vLLM allows aborting unknown ids silently

    outputs = [
        output
        async for output in engine.generate(
            "future prompt", SamplingParams(), "future"
        )
    ]

    assert outputs
    assert outputs[-1].finished is True


async def test_async_engine_runs_full_prepare_validation_before_stream():
    class PreparingRejectBackend:
        def __init__(self) -> None:
            self.fast_calls = 0
            self.full_calls = 0
            self.prepare_calls = 0
            self.stream_calls = 0

        def validate_request_before_prepare(self, request) -> None:
            self.fast_calls += 1

        def validate_request(self, request) -> None:
            self.full_calls += 1
            raise ValueError("full preparation rejected")

        async def prepare_request(self, request) -> None:
            self.prepare_calls += 1
            self.validate_request(request)

        async def stream(self, request):
            self.stream_calls += 1
            raise AssertionError("rejected request was dispatched")
            yield  # pragma: no cover

    backend = PreparingRejectBackend()
    engine = AsyncLLMEngine(backend=backend, model="custom")

    with pytest.raises(ValueError, match="full preparation rejected"):
        async for _ in engine.generate(
            "prompt",
            SamplingParams(max_tokens=1),
            "prepare-reject",
        ):
            pass

    assert (backend.fast_calls, backend.full_calls) == (1, 1)
    assert backend.prepare_calls == 1
    assert backend.stream_calls == 0
    assert engine._active == {}


async def test_abort_during_async_preparation_never_dispatches_stream():
    class GatedPreparingBackend:
        def __init__(self) -> None:
            self.entered = asyncio.Event()
            self.cancelled = False
            self.stream_calls = 0

        def validate_request_before_prepare(self, request) -> None:
            return None

        async def prepare_request(self, request) -> None:
            self.entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

        async def stream(self, request):
            self.stream_calls += 1
            raise AssertionError("aborted preparation was dispatched")
            yield  # pragma: no cover

    backend = GatedPreparingBackend()
    engine = AsyncLLMEngine(backend=backend, model="custom")

    async def collect():
        return [
            output
            async for output in engine.generate(
                "prompt",
                SamplingParams(max_tokens=1),
                "prepare-abort",
            )
        ]

    task = asyncio.create_task(collect())
    await backend.entered.wait()
    await engine.abort("prepare-abort")

    assert await task == []
    assert backend.cancelled is True
    assert backend.stream_calls == 0
    assert engine._active == {}


async def test_concurrent_generate_streams_are_isolated():
    import asyncio

    engine = _engine()

    async def collect(prompt: str, request_id: str) -> RequestOutput:
        async for output in engine.generate(prompt, SamplingParams(), request_id):
            last = output
        return last

    first, second = await asyncio.gather(
        collect("first prompt", "r1"), collect("second prompt", "r2")
    )
    assert first.request_id == "r1"
    assert second.request_id == "r2"
    assert first.outputs[0].text != second.outputs[0].text
