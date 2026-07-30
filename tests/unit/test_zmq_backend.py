"""Process-split backend parity: kairyu-proc ⇄ in-process kairyu (m8 D6).

One spawned service is shared module-wide (each child pays a full interpreter
import); tests end via the clean shutdown op so the child flushes coverage.
"""

import asyncio
import os
import signal

import pytest

from kairyu import SamplingParams
from kairyu.engine.backend import GenerationRequest
from kairyu.engine.core.engine_service import LEGACY_WIRE_VERSION, WIRE_VERSION
from kairyu.engine.kairyu_backend import KairyuBackend
from kairyu.engine.prompt import (
    MultimodalItem,
    MultimodalPrompt,
    PromptInput,
    TokensPrompt,
)
from kairyu.engine.registry import create_backend
from kairyu.engine.zmq_backend import ZmqEngineBackend

pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest.fixture(scope="module")
def zmq_backend():
    backend = ZmqEngineBackend(num_pages=256)
    yield backend
    asyncio.run(backend.shutdown())


def _request(request_id: str, prompt: PromptInput, **sampling) -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        prompt=prompt,
        sampling_params=SamplingParams(**sampling),
    )


async def test_generate_parity_with_in_process(zmq_backend):
    reference = await KairyuBackend(num_pages=256).generate(
        _request("p1", "parity across the process boundary", max_tokens=6)
    )
    result = await zmq_backend.generate(
        _request("p1", "parity across the process boundary", max_tokens=6)
    )
    ref = reference.completions[0]
    got = result.completions[0]
    assert got.token_ids == ref.token_ids  # sha256 tokenizer: process-stable
    assert got.text == ref.text
    assert got.finish_reason == ref.finish_reason == "length"


async def test_token_prompt_crosses_process_without_retokenization(zmq_backend):
    prompt = TokensPrompt(
        prompt_token_ids=(5, 8, 13, 21),
        prompt="display-only text",
    )
    params = {"max_tokens": 5, "temperature": 0}
    reference = await KairyuBackend(num_pages=256).generate(
        _request("token-wire", prompt, **params)
    )
    result = await zmq_backend.generate(
        _request("token-wire", prompt, **params)
    )

    assert result.prompt == prompt
    assert result.prompt_token_ids == prompt.prompt_token_ids
    assert result.usage is not None
    assert result.usage.prompt_tokens == len(prompt.prompt_token_ids)
    assert result.completions == reference.completions


async def test_multimodal_rejects_before_process_start():
    backend = ZmqEngineBackend(num_pages=64)
    request = _request(
        "mm-preflight",
        MultimodalPrompt(
            base="describe",
            items=(
                MultimodalItem(
                    modality="image",
                    encoding="uri",
                    data="https://example.test/image.png",
                ),
            ),
        ),
        max_tokens=1,
    )

    with pytest.raises(ValueError, match="does not support multimodal"):
        await backend.generate(request)
    assert backend._process is None
    assert backend._active_request_ids == set()


async def test_stream_yields_incremental_partials(zmq_backend):
    partials = []
    async for partial in zmq_backend.stream(_request("s1", "stream me please", max_tokens=5)):
        partials.append(partial)
    assert partials[-1].finished is True
    lengths = [len(p.completions[0].token_ids) for p in partials]
    assert lengths == sorted(lengths)
    assert lengths[-1] == 5


async def test_long_v2_generation_matches_in_process_with_logprobs(zmq_backend):
    request = _request(
        "long-v2",
        "long process wire parity",
        max_tokens=128,
        logprobs=3,
        ignore_eos=True,
    )
    reference = await KairyuBackend(num_pages=256).generate(request)
    result = await zmq_backend.generate(
        _request(
            "long-v2",
            "long process wire parity",
            max_tokens=128,
            logprobs=3,
            ignore_eos=True,
        )
    )
    expected = reference.completions[0]
    actual = result.completions[0]
    assert actual.token_ids == expected.token_ids
    assert actual.text == expected.text
    assert actual.finish_reason == expected.finish_reason == "length"
    assert actual.logprobs == expected.logprobs
    assert actual.logprob_content == expected.logprob_content
    assert actual.cumulative_logprob == expected.cumulative_logprob
    assert result.usage == reference.usage


async def test_raw_v2_events_are_one_snapshot_then_sequenced_deltas(zmq_backend):
    request = _request("raw-v2", "inspect delta frames", max_tokens=8)
    queue = await zmq_backend._submit(request)
    events = []
    try:
        while True:
            event = await queue.get()
            events.append(event)
            if event.get("finished"):
                break
    finally:
        zmq_backend._queues.pop(request.request_id, None)
        zmq_backend._release_wire_route(request.request_id)

    assert events[0]["event"] == "snapshot"
    assert [event["sequence"] for event in events] == list(range(len(events)))
    assert len({event["stream_id"] for event in events}) == 1
    assert all(event["event"] == "delta" for event in events[1:])
    assert all(
        {"outputs", "text", "logprobs", "logprob_content"}.isdisjoint(event)
        for event in events[1:]
    )


async def test_v2_delivery_fails_bad_stream_id_and_drops_retired_wire_generation():
    backend = ZmqEngineBackend(num_pages=64)
    queue = asyncio.Queue()
    backend._queues["delivery"] = queue
    backend._wire_request_ids["delivery"] = "wire-current"
    backend._public_request_ids["wire-current"] = "delivery"
    backend._stream_ids["delivery"] = "current"

    backend._deliver_event(
        {
            "wire_version": 2,
            "request_id": "wire-current",
            "event": "delta",
        }
    )
    malformed = queue.get_nowait()
    assert "omitted its stream_id" in malformed["error"]

    backend._deliver_event(
        {
            "wire_version": 2,
            "request_id": "wire-current",
            "stream_id": "stale",
            "event": "delta",
        }
    )
    malformed = queue.get_nowait()
    assert "stream_id mismatch" in malformed["error"]

    backend._deliver_event(
        {
            "wire_version": 2,
            "request_id": "wire-retired",
            "stream_id": "stale",
            "event": "delta",
        }
    )
    assert queue.empty()


async def test_new_service_keeps_legacy_client_cumulative_wire():
    backend = ZmqEngineBackend(num_pages=64)
    request = _request("legacy-client", "rolling upgrade", max_tokens=4)
    backend._wire_version = LEGACY_WIRE_VERSION
    try:
        queue = await backend._submit(request)
        events = []
        while True:
            event = await asyncio.wait_for(queue.get(), timeout=5)
            events.append(event)
            if event.get("finished"):
                break
        assert all("wire_version" not in event for event in events)
        assert all("outputs" in event and "text" in event for event in events)
        assert [len(event["outputs"]) for event in events] == sorted(
            len(event["outputs"]) for event in events
        )
    finally:
        backend._queues.pop(request.request_id, None)
        backend._release_wire_route(request.request_id)
        await backend.shutdown()


async def test_stop_string_works_across_process(zmq_backend):
    probe = await zmq_backend.generate(_request("probe", "stoppable text", max_tokens=8))
    text = probe.completions[0].text
    stop = text.split()[2]  # a mid-stream toy word
    result = await zmq_backend.generate(
        _request("s2", "stoppable text", max_tokens=8, stop=stop)
    )
    completion = result.completions[0]
    assert completion.finish_reason == "stop"
    assert stop not in completion.text


async def test_concurrent_requests(zmq_backend):
    results = await asyncio.gather(
        zmq_backend.generate(_request("c1", "first concurrent", max_tokens=4)),
        zmq_backend.generate(_request("c2", "second concurrent", max_tokens=4)),
        zmq_backend.generate(_request("c3", "third concurrent", max_tokens=4)),
    )
    assert all(r.finished for r in results)
    assert {r.request_id for r in results} == {"c1", "c2", "c3"}


async def test_stream_abandonment_sends_abort(zmq_backend):
    stream = zmq_backend.stream(_request("a1", "abandoned stream", max_tokens=16))
    first = None
    async for partial in stream:
        first = partial
        break  # client disconnects
    await stream.aclose()
    assert first is not None
    # the service keeps running and serves the next request fine
    follow_up = await zmq_backend.generate(_request("a2", "after abandon", max_tokens=3))
    assert follow_up.finished


async def test_usage_fields_cross_the_wire(zmq_backend):
    # second identical prompt: the radix cache serves the shared prefix
    # (>= one full 16-token page — the radix tree caches full pages only)
    prompt = " ".join(f"word{i}" for i in range(20))
    await zmq_backend.generate(_request("u1", prompt, max_tokens=4))
    queue = await zmq_backend._submit(_request("u2", prompt, max_tokens=4))
    events = []
    while True:
        event = await queue.get()
        events.append(event)
        if event.get("finished"):
            break
    zmq_backend._queues.pop("u2", None)
    zmq_backend._release_wire_route("u2")
    assert events[0]["num_prompt_tokens"] == 20
    assert events[-1]["num_cached_tokens"] >= 16


async def test_registered_as_kairyu_proc():
    backend = create_backend("kairyu-proc", num_pages=64)
    assert isinstance(backend, ZmqEngineBackend)


async def test_rejects_non_string_tokenizer():
    from kairyu.engine.tokenizer import ToyTokenizer

    with pytest.raises(ValueError, match="string tokenizer"):
        ZmqEngineBackend(tokenizer=ToyTokenizer())  # type: ignore[arg-type]


async def test_cuda_graph_serving_options_cross_the_process_boundary():
    backend = ZmqEngineBackend(
        model_path="/models/tiny",
        pipeline_depth=2,
        decode_mode="cuda_graph",
        cuda_graph_max_batch=4,
        cuda_graph_max_pages=32,
        cuda_graph_warmup_iters=1,
    )

    assert backend._config["pipeline_depth"] == 2
    assert backend._config["decode_mode"] == "cuda_graph"
    assert backend._config["cuda_graph_max_batch"] == 4
    assert backend._config["cuda_graph_max_pages"] == 32
    assert backend._config["cuda_graph_warmup_iters"] == 1


async def test_inflight_request_sees_service_death():
    # A request awaiting when the child dies must be delivered an error event
    # (death detection), not hang forever.
    backend = ZmqEngineBackend(num_pages=64, death_timeout_s=2.0)
    try:
        await backend.generate(_request("d1", "warm up", max_tokens=2))
        queue = await backend._submit(_request("d2", "in flight", max_tokens=64))
        backend._process.kill()  # crash while d2 awaits
        event = await asyncio.wait_for(queue.get(), timeout=10)
        assert "error" in event  # delivered, not a permanent hang
    finally:
        backend._queues.pop("d2", None)
        backend._release_wire_route("d2")
        await backend.shutdown()


async def test_backend_recovers_after_service_death():
    # E1: once the dead child is observed, the backend must respawn a fresh
    # engine service for later requests instead of leaving them to hang.
    backend = ZmqEngineBackend(num_pages=64, death_timeout_s=2.0)
    try:
        await backend.generate(_request("r1", "warm up", max_tokens=2))
        queue = await backend._submit(_request("r2", "in flight", max_tokens=64))
        backend._process.kill()
        await asyncio.wait_for(queue.get(), timeout=10)  # error event; receiver exits
        backend._queues.pop("r2", None)
        backend._release_wire_route("r2")
        # the next request respawns a fresh child and completes normally
        result = await asyncio.wait_for(
            backend.generate(_request("r3", "recovered", max_tokens=2)), timeout=15
        )
        assert result.completions[0].token_ids
    finally:
        await backend.shutdown()


async def test_shutdown_is_clean_and_idempotent():
    backend = ZmqEngineBackend(num_pages=64)
    await backend.generate(_request("x1", "before shutdown", max_tokens=2))
    process = backend._process
    await backend.shutdown()
    assert process is not None and not process.is_alive()
    assert process.exitcode == 0  # clean exit: coverage flushed
    await backend.shutdown()  # idempotent


async def test_duplicate_request_id_preserves_original_queue_and_can_be_reused(
    zmq_backend,
):
    original = asyncio.create_task(
        zmq_backend.generate(_request("same", "first", max_tokens=10_000))
    )
    for _ in range(500):
        if "same" in zmq_backend._active_request_ids and "same" in zmq_backend._queues:
            break
        await asyncio.sleep(0.01)
    assert "same" in zmq_backend._active_request_ids
    assert "same" in zmq_backend._queues
    original_queue = zmq_backend._queues["same"]

    with pytest.raises(ValueError, match="duplicate request_id"):
        await zmq_backend.generate(_request("same", "second", max_tokens=2))
    duplicate_stream = zmq_backend.stream(
        _request("same", "second stream", max_tokens=2)
    )
    with pytest.raises(ValueError, match="duplicate request_id"):
        await anext(duplicate_stream)
    assert zmq_backend._queues["same"] is original_queue

    original.cancel()
    with pytest.raises(asyncio.CancelledError):
        await original
    assert "same" not in zmq_backend._active_request_ids
    assert "same" not in zmq_backend._queues

    reused = await zmq_backend.generate(_request("same", "reused", max_tokens=2))
    assert reused.finished is True


@pytest.mark.parametrize("wire_version", (LEGACY_WIRE_VERSION, WIRE_VERSION))
async def test_cancelled_request_id_can_be_reused_after_queued_abort(wire_version):
    sigstop = getattr(signal, "SIGSTOP", None)
    sigcont = getattr(signal, "SIGCONT", None)
    if sigstop is None or sigcont is None:
        pytest.skip("requires POSIX process stop/continue signals")

    backend = ZmqEngineBackend(num_pages=256)
    backend._wire_version = wire_version
    original = None
    reused = None
    process = None
    stopped = False
    try:
        original = asyncio.create_task(
            backend.generate(_request("queued-reuse", "first", max_tokens=10_000))
        )
        for _ in range(500):
            if (
                "queued-reuse" in backend._active_request_ids
                and "queued-reuse" in backend._queues
                and backend._process is not None
            ):
                break
            await asyncio.sleep(0.01)
        assert "queued-reuse" in backend._active_request_ids
        assert "queued-reuse" in backend._queues
        process = backend._process
        assert process is not None and process.is_alive()

        os.kill(process.pid, sigstop)
        stopped = True
        waited_pid, status = await asyncio.to_thread(
            os.waitpid, process.pid, os.WUNTRACED
        )
        assert waited_pid == process.pid
        assert os.WIFSTOPPED(status)
        original.cancel()
        with pytest.raises(asyncio.CancelledError):
            await original

        reused = asyncio.create_task(
            backend.generate(_request("queued-reuse", "reused", max_tokens=2))
        )
        for _ in range(500):
            if (
                "queued-reuse" in backend._active_request_ids
                and "queued-reuse" in backend._queues
            ):
                break
            await asyncio.sleep(0.01)
        assert "queued-reuse" in backend._active_request_ids
        assert "queued-reuse" in backend._queues

        os.kill(process.pid, sigcont)
        stopped = False
        result = await asyncio.wait_for(reused, timeout=15)
        assert result.finished is True
    finally:
        try:
            if stopped and process is not None:
                try:
                    os.kill(process.pid, sigcont)
                except ProcessLookupError:
                    pass
        finally:
            try:
                for task in (original, reused):
                    if task is not None and not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
            finally:
                await backend.shutdown()


async def test_pipeline_depth_three_cancelled_id_can_be_reused_immediately():
    backend = ZmqEngineBackend(num_pages=256, pipeline_depth=3)
    stream = backend.stream(
        _request(
            "pipeline-reuse",
            "first pipelined request",
            max_tokens=1_000,
            ignore_eos=True,
        )
    )
    try:
        first = await asyncio.wait_for(anext(stream), timeout=15)
        assert first.finished is False
        await stream.aclose()
        result = await asyncio.wait_for(
            backend.generate(
                _request(
                    "pipeline-reuse",
                    "reused pipelined request",
                    max_tokens=3,
                    ignore_eos=True,
                )
            ),
            timeout=15,
        )
        assert result.finished is True
        assert len(result.completions[0].token_ids) == 3
    finally:
        await stream.aclose()
        await backend.shutdown()


async def test_submit_failure_clears_request_reservation_and_queue(monkeypatch):
    class FailingSocket:
        async def send(self, _payload):
            raise RuntimeError("send failed")

    backend = ZmqEngineBackend(num_pages=64)

    async def already_started():
        return None

    monkeypatch.setattr(backend, "_ensure_started", already_started)
    backend._socket = FailingSocket()

    with pytest.raises(RuntimeError, match="send failed"):
        await backend.generate(_request("send-failure", "prompt", max_tokens=2))

    assert "send-failure" not in backend._active_request_ids
    assert "send-failure" not in backend._queues


@pytest.mark.parametrize("original_api", ["generate", "stream"])
async def test_duplicate_request_id_stays_reserved_until_abort_finishes(
    monkeypatch, original_api
):
    backend = ZmqEngineBackend(num_pages=64)
    first_queue = asyncio.Queue()
    abort_started = asyncio.Event()
    finish_abort = asyncio.Event()

    async def controlled_submit(request):
        if request.prompt == "first":
            backend._queues[request.request_id] = first_queue
            return first_queue
        if request.prompt == "reused":
            queue = asyncio.Queue()
            backend._queues[request.request_id] = queue
            queue.put_nowait(
                {
                    "text": "done",
                    "outputs": [1],
                    "finished": True,
                    "finish_reason": "length",
                }
            )
            return queue
        raise AssertionError("duplicate request reached submit")

    async def controlled_abort(_request_id):
        abort_started.set()
        await finish_abort.wait()

    monkeypatch.setattr(backend, "_submit", controlled_submit)
    monkeypatch.setattr(backend, "_abort", controlled_abort)

    request = _request("abort-race", "first", max_tokens=64)
    if original_api == "generate":
        original = asyncio.create_task(backend.generate(request))
    else:
        original_stream = backend.stream(request)
        original = asyncio.create_task(anext(original_stream))

    for _ in range(100):
        if "abort-race" in backend._active_request_ids:
            break
        await asyncio.sleep(0.01)
    assert "abort-race" in backend._active_request_ids
    original.cancel()
    await asyncio.wait_for(abort_started.wait(), timeout=1)

    try:
        with pytest.raises(ValueError, match="duplicate request_id"):
            await backend.generate(
                _request("abort-race", "duplicate generate", max_tokens=2)
            )
        duplicate_stream = backend.stream(
            _request("abort-race", "duplicate stream", max_tokens=2)
        )
        with pytest.raises(ValueError, match="duplicate request_id"):
            await anext(duplicate_stream)
        assert "abort-race" in backend._active_request_ids
    finally:
        finish_abort.set()
        try:
            await original
        except asyncio.CancelledError:
            pass

    assert original.cancelled()
    assert "abort-race" not in backend._active_request_ids
    reused = await backend.generate(_request("abort-race", "reused", max_tokens=2))
    assert reused.finished is True
