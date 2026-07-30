"""Contract tests for vLLM AsyncLLMEngine-style usage (design doc D2).

Mirrors the vLLM pattern:
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(model=...))
    async for output in engine.generate(prompt, params, request_id): ...
"""

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
