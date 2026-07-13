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


async def test_duplicate_request_id_is_rejected_without_disrupting_active_request():
    backend = KairyuBackend(num_pages=256, runner=_SlowRunner())
    first = asyncio.create_task(
        backend.generate(_request("duplicate", "first", 10_000))
    )
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
    blocker = asyncio.create_task(
        backend.generate(_request("atomic#c1", "blocker", 10_000))
    )
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


async def test_multi_stream_partial_submit_failure_rolls_back_all_sub_ids():
    backend = KairyuBackend(num_pages=256)
    real_submit = backend._loop.submit
    submit_count = 0

    def fail_second_submit(request_id, prompt, params):
        nonlocal submit_count
        submit_count += 1
        if submit_count == 2:
            raise RuntimeError("injected submit failure")
        real_submit(request_id, prompt, params)

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

    def fail_submit(request_id, prompt, params):
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
        backend._queues[request_id] is queue
        for request_id, queue in original_queues.items()
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
