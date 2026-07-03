"""Process-split backend parity: kairyu-proc ⇄ in-process kairyu (m8 D6).

One spawned service is shared module-wide (each child pays a full interpreter
import); tests end via the clean shutdown op so the child flushes coverage.
"""

import asyncio

import pytest

from kairyu import SamplingParams
from kairyu.engine.backend import GenerationRequest
from kairyu.engine.kairyu_backend import KairyuBackend
from kairyu.engine.registry import create_backend
from kairyu.engine.zmq_backend import EngineServiceError, ZmqEngineBackend

pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest.fixture(scope="module")
def zmq_backend():
    backend = ZmqEngineBackend(num_pages=256)
    yield backend
    asyncio.run(backend.shutdown())


def _request(request_id: str, prompt: str, **sampling) -> GenerationRequest:
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


async def test_stream_yields_incremental_partials(zmq_backend):
    partials = []
    async for partial in zmq_backend.stream(_request("s1", "stream me please", max_tokens=5)):
        partials.append(partial)
    assert partials[-1].finished is True
    lengths = [len(p.completions[0].token_ids) for p in partials]
    assert lengths == sorted(lengths)
    assert lengths[-1] == 5


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
    assert events[0]["num_prompt_tokens"] == 20
    assert events[-1]["num_cached_tokens"] >= 16


async def test_registered_as_kairyu_proc():
    backend = create_backend("kairyu-proc", num_pages=64)
    assert isinstance(backend, ZmqEngineBackend)


async def test_rejects_non_string_tokenizer():
    from kairyu.engine.tokenizer import ToyTokenizer

    with pytest.raises(ValueError, match="string tokenizer"):
        ZmqEngineBackend(tokenizer=ToyTokenizer())  # type: ignore[arg-type]


async def test_service_death_surfaces_as_error():
    backend = ZmqEngineBackend(num_pages=64, death_timeout_s=2.0)
    try:
        await backend.generate(_request("d1", "warm up", max_tokens=2))
        backend._process.kill()  # simulate a crashed engine service
        with pytest.raises(EngineServiceError):
            await asyncio.wait_for(
                backend.generate(_request("d2", "after crash", max_tokens=2)), timeout=10
            )
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
