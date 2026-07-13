import asyncio
import threading
import time
from collections.abc import Mapping

import httpx
import pytest

from kairyu import SamplingParams
from kairyu.engine.backend import GenerationRequest
from kairyu.engine.core.sampling_types import SampledToken
from kairyu.engine.core.scheduler import ScheduledChunk
from kairyu.engine.kairyu_backend import KairyuBackend
from kairyu.engine.registry import create_backend
from kairyu.entrypoints.server.app import create_app


def _request(request_id: str, prompt: str, max_tokens: int = 4) -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        prompt=prompt,
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )


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
    assert len(partials) >= 5
    assert partials[-1].finished is True
    assert all(p.finished is False for p in partials[:-1])
    lengths = [len(p.completions[0].token_ids) for p in partials]
    assert lengths == sorted(lengths)  # monotonically growing


async def test_stream_abandonment_aborts_engine_work():
    backend = KairyuBackend(num_pages=256, runner=_SlowRunner())
    stream = backend.stream(
        _request("abandoned", "keep generating", max_tokens=10_000)
    )
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


async def test_pump_failure_purges_requests_and_backend_recovers():
    backend = KairyuBackend(num_pages=256, runner=_FailOnceRunner())
    with pytest.raises(RuntimeError, match="injected runner failure"):
        await backend.generate(_request("failed", "first", 2))

    assert backend._loop.has_work() is False
    assert "failed" not in backend._scheduler.states
    recovered = await backend.generate(_request("recovered", "second", 2))
    assert recovered.finished


async def test_request_submitted_during_failure_purge_is_pumped():
    backend = KairyuBackend(num_pages=256, runner=_FailOnceRunner())
    purge_started = threading.Event()
    allow_purge = threading.Event()
    original_purge = backend._loop.purge

    def delayed_purge(request_ids: tuple[str, ...]) -> None:
        purge_started.set()
        if not allow_purge.wait(timeout=2):
            raise TimeoutError("test did not release purge")
        original_purge(request_ids)

    backend._loop.purge = delayed_purge
    failed = asyncio.create_task(backend.generate(_request("failed", "first", 2)))
    assert await asyncio.to_thread(purge_started.wait, 2)

    recovered = asyncio.create_task(
        backend.generate(_request("submitted-during-purge", "second", 2))
    )
    await asyncio.sleep(0)
    allow_purge.set()

    with pytest.raises(RuntimeError, match="injected runner failure"):
        await failed
    result = await asyncio.wait_for(recovered, timeout=2)
    assert result.finished


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


async def test_full_stack_openai_server_over_engine_core():
    app = create_app(engines={"kairyu-cpu": KairyuBackend(num_pages=256)})
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
