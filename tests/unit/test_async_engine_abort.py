import asyncio
import contextlib
from collections import Counter, defaultdict

import pytest

from kairyu import AsyncLLMEngine, SamplingParams
from kairyu.engine.backend import GenerationRequest, GenerationResult
from kairyu.outputs import CompletionOutput


class ControlledBackend:
    def __init__(self) -> None:
        self.started: defaultdict[str, asyncio.Event] = defaultdict(asyncio.Event)
        self.release: defaultdict[str, asyncio.Event] = defaultdict(asyncio.Event)
        self.closed: Counter[str] = Counter()

    async def stream(self, request: GenerationRequest):
        try:
            self.started[request.request_id].set()
            await self.release[request.request_id].wait()
            for index in range(2):
                yield GenerationResult(
                    request_id=request.request_id,
                    prompt=request.prompt,
                    completions=(
                        CompletionOutput(
                            index=0,
                            text=f"{request.request_id}-{index}",
                            token_ids=(index,),
                            finish_reason="stop" if index else None,
                        ),
                    ),
                    finished=index == 1,
                )
        finally:
            self.closed[request.request_id] += 1

    async def shutdown(self) -> None:
        return None


class RaisingStreamBackend:
    def stream(self, request: GenerationRequest):
        raise RuntimeError(f"cannot stream {request.request_id}")

    async def shutdown(self) -> None:
        return None


async def _collect(engine: AsyncLLMEngine, request_id: str):
    return [
        output
        async for output in engine.generate(
            f"prompt-{request_id}", SamplingParams(), request_id
        )
    ]


async def test_abort_interrupts_blocked_stream_and_closes_backend_once():
    backend = ControlledBackend()
    engine = AsyncLLMEngine(backend)
    stream = engine.generate("blocked prompt", SamplingParams(), "blocked")
    pending = asyncio.create_task(anext(stream))
    await backend.started["blocked"].wait()

    await engine.abort("blocked")

    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(pending, timeout=1)
    assert backend.closed["blocked"] == 1
    assert engine._active == {}


async def test_inactive_aborts_retain_no_state_and_normal_completion_deregisters():
    backend = ControlledBackend()
    engine = AsyncLLMEngine(backend)

    for index in range(1_000):
        await engine.abort(f"unknown-{index}")
    assert engine._active == {}

    backend.release["normal"].set()
    outputs = await _collect(engine, "normal")

    assert outputs[-1].finished is True
    assert engine._active == {}
    assert backend.closed["normal"] == 1


async def test_stream_creation_failure_deregisters_request_id():
    engine = AsyncLLMEngine(RaisingStreamBackend())
    stream = engine.generate("failed prompt", SamplingParams(), "failed")

    with pytest.raises(RuntimeError, match="cannot stream failed"):
        await anext(stream)

    assert engine._active == {}


async def test_abort_is_isolated_and_finished_request_id_can_be_reused():
    backend = ControlledBackend()
    engine = AsyncLLMEngine(backend)
    stopped = asyncio.create_task(_collect(engine, "stop"))
    kept = asyncio.create_task(_collect(engine, "keep"))
    await asyncio.gather(
        backend.started["stop"].wait(), backend.started["keep"].wait()
    )

    await engine.abort("stop")
    backend.release["keep"].set()
    stopped_outputs, kept_outputs = await asyncio.wait_for(
        asyncio.gather(stopped, kept), timeout=1
    )

    assert stopped_outputs == []
    assert kept_outputs[-1].finished is True
    assert backend.closed == Counter({"stop": 1, "keep": 1})

    backend.release["stop"].set()
    reused_outputs = await _collect(engine, "stop")
    assert reused_outputs[-1].finished is True
    assert backend.closed["stop"] == 2
    assert engine._active == {}


async def test_duplicate_active_id_fails_without_deregistering_original():
    backend = ControlledBackend()
    engine = AsyncLLMEngine(backend)
    original = engine.generate("original", SamplingParams(), "duplicate")
    pending = asyncio.create_task(anext(original))
    await backend.started["duplicate"].wait()

    try:
        duplicate = engine.generate("second", SamplingParams(), "duplicate")
        with pytest.raises(ValueError, match="already active"):
            await asyncio.wait_for(anext(duplicate), timeout=1)
        assert "duplicate" in engine._active
    finally:
        await engine.abort("duplicate")
        backend.release["duplicate"].set()
        with contextlib.suppress(StopAsyncIteration):
            await asyncio.wait_for(pending, timeout=1)
    assert engine._active == {}
    assert backend.closed["duplicate"] == 1


async def test_consumer_close_deregisters_and_closes_backend_once():
    backend = ControlledBackend()
    engine = AsyncLLMEngine(backend)
    backend.release["consumer"].set()
    stream = engine.generate("consumer prompt", SamplingParams(), "consumer")

    await anext(stream)
    assert "consumer" in engine._active
    await stream.aclose()

    assert engine._active == {}
    assert backend.closed["consumer"] == 1
