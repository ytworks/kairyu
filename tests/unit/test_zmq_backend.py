"""Process-split backend parity: kairyu-proc ⇄ in-process kairyu (m8 D6).

One spawned service is shared module-wide (each child pays a full interpreter
import); tests end via the clean shutdown op so the child flushes coverage.
"""

import asyncio
import contextlib
import gc
import multiprocessing
import os
import queue as queue_module
import signal
import sys
import threading
import time
import types
import weakref
from pathlib import Path

import httpx
import msgpack
import pytest

from kairyu import SamplingParams
from kairyu.engine.backend import GenerationRequest, GenerationStageMetric
from kairyu.engine.core.attention_selector import AttentionBackendDecision
from kairyu.engine.core.engine_service import (
    LEGACY_WIRE_VERSION,
    STARTUP_WIRE_VERSION,
    WIRE_VERSION,
    _close_engine_service_loop,
    _send_optional_startup_frame,
    _startup_ready_frame,
    run_engine_service,
)
from kairyu.engine.core.sampling_types import SampledToken
from kairyu.engine.kairyu_backend import KairyuBackend, build_engine_loop
from kairyu.engine.prompt import (
    MultimodalItem,
    MultimodalPrompt,
    PromptInput,
    TemplatedPrompt,
    TextPrompt,
    TokensPrompt,
)
from kairyu.engine.registry import create_backend
from kairyu.engine.tokenizer import ToyTokenizer
from kairyu.engine.zmq_backend import (
    EngineServiceError,
    ZmqEngineBackend,
    _decision_from_startup_frame,
    _decode_graph_metadata_from_wire,
    _ensure_child_subreaper,
    _generation_defaults_from_startup_frame,
    _kill_and_reap_process,
    _process_group_member_states,
    _run_engine_service_in_owned_process_group,
    _tensor_parallel_size_from_startup_frame,
    _validate_startup_tensor_parallel_identity,
)
from kairyu.entrypoints.server.app import create_app
from kairyu.models.generation import GenerationDefaults
from kairyu.sampling_params import GENERATION_CONFIG_SAMPLING_FIELDS

pytestmark = pytest.mark.asyncio(loop_scope="module")


async def test_constructor_preserves_legacy_tokenizer_positional_slot():
    backend = ZmqEngineBackend(4096, 16, 2048, 256, 60.0, "toy")

    assert backend._config["tokenizer"] == "toy"
    assert backend._config["tensor_parallel_size"] == 1
    assert backend._config["max_num_partial_prefills"] == 2


async def test_constructor_forwards_partial_prefill_limit_to_child_config():
    backend = ZmqEngineBackend(max_num_partial_prefills=7, max_num_seqs=13)

    assert backend._config["max_num_partial_prefills"] == 7
    assert backend.sequence_budget == 13


async def test_validation_key_uses_effective_tokenizer_and_context_contract():
    class StatefulValidationBackend(ZmqEngineBackend):
        pass

    first = ZmqEngineBackend(num_pages=64, max_model_len=32)
    equivalent = ZmqEngineBackend(
        num_pages=32,
        tokenizer="toy",
        max_model_len=32,
    )
    different_limit = ZmqEngineBackend(num_pages=64, max_model_len=64)
    different_tokenizer = ZmqEngineBackend(
        num_pages=64,
        tokenizer="other-tokenizer",
        max_model_len=32,
    )
    different_model = ZmqEngineBackend(
        num_pages=64,
        tokenizer="toy",
        model_path="other-model",
        max_model_len=32,
    )
    subclass = StatefulValidationBackend(num_pages=64, max_model_len=32)

    assert first.request_validation_key == (None, "toy", 32)
    assert first.request_validation_key == equivalent.request_validation_key
    assert first.request_validation_key != different_limit.request_validation_key
    assert first.request_validation_key != different_tokenizer.request_validation_key
    assert first.request_validation_key != different_model.request_validation_key
    assert subclass.request_validation_key is None
    assert isinstance(hash(first.request_validation_key), int)


_READY_DECISION = {
    "requested": "flashattention4",
    "resolved": "flashattention4",
    "source": "env",
    "components": {
        "prefill": "flashattention4",
        "decode": "flashinfer",
        "kv_mode": "paged-materialized",
    },
    "rationale": "constructed in child",
    "architecture": {
        "arch": "cuda",
        "device_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "sm": 120,
        "kernel_tier": "full",
    },
}

_READY_GENERATION_DEFAULTS = {
    "eos_token_id": 151645,
    "stop_token_ids": [151643],
    "sampling": {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.05,
        "repetition_penalty": 1.1,
    },
    "mode": "auto",
    "source": "generation_config.json",
}


def _fake_ready_engine_service(port_pipe, _config):
    port_pipe.send(43210)
    port_pipe.send(
        {
            "startup_wire_version": STARTUP_WIRE_VERSION,
            "status": "ready",
            "attention_backend_decision": _READY_DECISION,
            "kv_cache_dtype_requested": "auto",
            "kv_cache_dtype_resolved": "not-applicable",
            "generation_defaults": _READY_GENERATION_DEFAULTS,
        }
    )
    port_pipe.close()
    time.sleep(60)


def _fake_legacy_engine_service(port_pipe, _config):
    port_pipe.send(43211)
    port_pipe.close()
    time.sleep(60)


def _fake_generation_mode_mismatch_engine_service(port_pipe, _config):
    port_pipe.send(43215)
    port_pipe.send(
        {
            "startup_wire_version": STARTUP_WIRE_VERSION,
            "status": "ready",
            "attention_backend_decision": None,
            "generation_defaults": {
                **_READY_GENERATION_DEFAULTS,
                "mode": "vllm",
                "source": "vllm",
            },
        }
    )
    port_pipe.close()
    time.sleep(60)


def _fake_failing_engine_service(port_pipe, _config):
    port_pipe.send(43212)
    port_pipe.send(
        {
            "startup_wire_version": STARTUP_WIRE_VERSION,
            "status": "error",
            "error_type": "RuntimeError",
            "error": "unsupported SM",
        }
    )
    port_pipe.close()


def _fake_slow_starting_engine_service(port_pipe, _config):
    port_pipe.send(43213)
    _config["_test_port_sent"].set()
    time.sleep(60)


def _fake_max_model_len_engine_service(port_pipe, config):
    port_pipe.send(43214)
    if config.get("max_model_len") != 8192:
        port_pipe.send(
            {
                "startup_wire_version": STARTUP_WIRE_VERSION,
                "status": "error",
                "error_type": "AssertionError",
                "error": "max_model_len did not cross the spawn boundary",
            }
        )
        port_pipe.close()
        return
    port_pipe.send(
        {
            "startup_wire_version": STARTUP_WIRE_VERSION,
            "status": "ready",
            "attention_backend_decision": None,
        }
    )
    port_pipe.close()
    time.sleep(60)


def _fake_rank_worker(ignore_sigterm=False, ready_pipe=None):
    if ignore_sigterm:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if ready_pipe is not None:
        ready_pipe.send_bytes(b"ready")
        ready_pipe.close()
    time.sleep(60)


def _fake_nested_engine_service(port_pipe, config):
    spawn = multiprocessing.get_context("spawn")
    rank_ready_recv, rank_ready_send = spawn.Pipe(duplex=False)
    rank = spawn.Process(
        target=_fake_rank_worker,
        args=(
            config.get("_test_rank_ignore_sigterm", False),
            rank_ready_send,
        ),
    )
    rank.start()
    rank_ready_send.close()
    if not rank_ready_recv.poll(timeout=5):
        raise RuntimeError("fake rank did not become ready")
    rank_ready_recv.recv_bytes()
    rank_ready_recv.close()
    config["_test_rank_pid_pipe"].send(rank.pid)
    config["_test_rank_pid_pipe"].close()
    port_pipe.send(43216)
    port_pipe.send(
        {
            "startup_wire_version": STARTUP_WIRE_VERSION,
            "status": "ready",
            "tensor_parallel_size": 2,
        }
    )
    port_pipe.close()
    time.sleep(60)


def _fake_api_parent_with_nested_service(result_pipe):
    spawn = multiprocessing.get_context("spawn")
    port_parent, port_child = spawn.Pipe()
    child_lifetime, parent_lifetime = spawn.Pipe(duplex=False)
    rank_pid_recv, rank_pid_send = spawn.Pipe(duplex=False)
    service = spawn.Process(
        target=_run_engine_service_in_owned_process_group,
        args=(
            _fake_nested_engine_service,
            child_lifetime,
            port_child,
            {"_test_rank_pid_pipe": rank_pid_send},
        ),
    )
    service.start()
    child_lifetime.close()
    port_child.close()
    rank_pid_send.close()
    port_parent.recv()
    port_parent.recv()
    if not rank_pid_recv.poll(timeout=5):
        raise RuntimeError("fake service did not report its rank PID")
    rank_pid = rank_pid_recv.recv()
    rank_pid_recv.close()
    result_pipe.send((service.pid, rank_pid))
    result_pipe.close()
    # Keep the send-only lease alive. An external SIGKILL of this process
    # closes it in the kernel and the service watchdog kills the whole group.
    time.sleep(60)
    parent_lifetime.close()


class _NeverSendingSocket:
    def __init__(self) -> None:
        self.send_started = False
        self.send_cancelled = False
        self.closed = False

    async def send(self, _payload):
        self.send_started = True
        try:
            await asyncio.Future()
        finally:
            self.send_cancelled = True

    def close(self, *, linger):
        assert linger == 0
        self.closed = True


class _FakeServicePortPipe:
    def __init__(self) -> None:
        self.frames = []
        self.closed = False

    def send(self, frame):
        self.frames.append(frame)

    def close(self):
        self.closed = True


class _FakeServiceSocket:
    def __init__(self, message=None) -> None:
        self.message = message
        self.sent = []
        self.closed = False

    def bind_to_random_port(self, endpoint):
        assert endpoint == "tcp://127.0.0.1"
        return 43123

    def poll(self, _timeout):
        if self.message is not None:
            return True
        raise AssertionError("engine service continued polling after a fatal frame")

    def recv_multipart(self):
        assert self.message is not None
        message = self.message
        self.message = None
        return [b"client", msgpack.packb(message)]

    def send_multipart(self, frames):
        self.sent.append(frames)

    def close(self, *, linger):
        assert linger == 0
        self.closed = True


class _QueuedServiceSocket:
    """Thread-safe fake ROUTER used to exercise the live service state machine."""

    def __init__(self) -> None:
        self._incoming: queue_module.Queue[tuple[bytes, bytes]] = (
            queue_module.Queue()
        )
        self._staged: tuple[bytes, bytes] | None = None
        self.sent: list[list[bytes]] = []
        self.send_threads: list[int] = []
        self.terminal_sent = threading.Event()
        self.shutdown_on_terminal = False
        self.closed = False

    def bind_to_random_port(self, endpoint):
        assert endpoint == "tcp://127.0.0.1"
        return 43124

    def enqueue(self, message: dict, identity: bytes = b"client") -> None:
        self._incoming.put((identity, msgpack.packb(message)))

    def poll(self, timeout):
        if self._staged is not None:
            return True
        try:
            self._staged = self._incoming.get(timeout=max(0, timeout) / 1000)
        except queue_module.Empty:
            return False
        return True

    def recv_multipart(self):
        assert self._staged is not None
        staged, self._staged = self._staged, None
        return list(staged)

    def send_multipart(self, frames):
        self.send_threads.append(threading.get_ident())
        self.sent.append(frames)
        payload = msgpack.unpackb(frames[1])
        if payload.get("request_id") and payload.get("finished"):
            self.terminal_sent.set()
            if self.shutdown_on_terminal:
                self.enqueue({"op": "shutdown"}, identity=b"shutdown")

    def close(self, *, linger):
        assert linger == 0
        self.closed = True


class _FakeServiceContext:
    def __init__(self, service_socket) -> None:
        self.service_socket = service_socket
        self.terminated = False

    def socket(self, _kind):
        return self.service_socket

    def term(self):
        self.terminated = True


def _install_fake_service_zmq(monkeypatch, service_socket):
    context = _FakeServiceContext(service_socket)
    fake_zmq = types.ModuleType("zmq")
    fake_zmq.ROUTER = object()
    fake_zmq.Context = lambda: context
    monkeypatch.setitem(sys.modules, "zmq", fake_zmq)
    return context


@pytest.fixture(scope="module")
def zmq_backend():
    backend = ZmqEngineBackend(num_pages=256)
    yield backend
    asyncio.run(backend.shutdown())


def _request(
    request_id: str,
    prompt: PromptInput,
    *,
    trace_requested: bool = False,
    **sampling,
) -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        prompt=prompt,
        sampling_params=SamplingParams(**sampling),
        trace_requested=trace_requested,
    )


class _CountingTokenizer(ToyTokenizer):
    def __init__(self) -> None:
        self.encoded: list[str] = []
        self.encode_threads: list[int] = []

    def encode(self, text: str) -> tuple[int, ...]:
        self.encoded.append(text)
        self.encode_threads.append(threading.get_ident())
        return super().encode(text)


class _GatedTokenizer(ToyTokenizer):
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def encode(self, text: str) -> tuple[int, ...]:
        self.calls += 1
        self.entered.set()
        if not self.release.wait(2):
            raise TimeoutError("test did not release parent tokenization")
        return super().encode(text)


class _OverlapRejectingTokenizer(ToyTokenizer):
    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._active = False
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def encode(self, text: str) -> tuple[int, ...]:
        with self._guard:
            if self._active:
                raise RuntimeError("concurrent encode")
            self._active = True
        try:
            self.calls += 1
            self.entered.set()
            if not self.release.wait(2):
                raise TimeoutError("test did not release parent tokenization")
            return super().encode(text)
        finally:
            with self._guard:
                self._active = False


@pytest.mark.parametrize("max_model_len", [None, 32])
async def test_direct_public_calls_parent_tokenize_off_loop_once_each(
    max_model_len,
):
    tokenizer = _CountingTokenizer()
    backend = ZmqEngineBackend(
        num_pages=64,
        max_model_len=max_model_len,
    )
    backend._preflight_tokenizer = tokenizer
    event_loop_thread = threading.get_ident()
    try:
        generated = await backend.generate(
            _request("direct-generate", "direct generate", max_tokens=1)
        )
        streamed = [
            partial
            async for partial in backend.stream(
                _request("direct-stream", "direct stream", max_tokens=1)
            )
        ][-1]

        assert generated.finished
        assert streamed.finished
        assert tokenizer.encoded == ["direct generate", "direct stream"]
        assert len(tokenizer.encode_threads) == 2
        assert all(
            thread_id != event_loop_thread
            for thread_id in tokenizer.encode_threads
        )
        assert backend._prepared_requests == {}
    finally:
        await backend.shutdown()


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
    assert reference.stage_metrics == result.stage_metrics == ()


async def test_trace_stage_metric_schema_matches_in_process(zmq_backend):
    reference = await KairyuBackend(num_pages=256).generate(
        _request(
            "trace-inproc",
            "trace schema parity",
            max_tokens=4,
            trace_requested=True,
        )
    )
    result = await zmq_backend.generate(
        _request(
            "trace-proc",
            "trace schema parity",
            max_tokens=4,
            trace_requested=True,
        )
    )

    assert reference.stage_metrics
    assert result.stage_metrics
    for metrics in (reference.stage_metrics, result.stage_metrics):
        assert type(metrics) is tuple
        assert all(type(metric) is GenerationStageMetric for metric in metrics)
        assert len({metric.stage for metric in metrics}) == len(metrics)
        assert all(metric.stage.strip() for metric in metrics)
        assert all(metric.duration_ns >= 0 for metric in metrics)
        assert all(metric.occurrences >= 1 for metric in metrics)
    assert {metric.stage for metric in result.stage_metrics} == {
        metric.stage for metric in reference.stage_metrics
    }


async def test_token_prompt_crosses_process_without_retokenization(zmq_backend):
    prompt = TokensPrompt(
        prompt_token_ids=(5, 8, 13, 21),
        prompt="display-only text",
    )
    params = {"max_tokens": 5, "temperature": 0}
    reference = await KairyuBackend(num_pages=256).generate(
        _request("token-wire", prompt, **params)
    )
    result = await zmq_backend.generate(_request("token-wire", prompt, **params))

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
        {"outputs", "text", "logprobs", "logprob_content"}.isdisjoint(event) for event in events[1:]
    )
    assert all("stage_metrics" not in event for event in events)


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
    request = _request(
        "legacy-client",
        "rolling upgrade",
        max_tokens=4,
        trace_requested=True,
    )
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
        assert all("stage_metrics" not in event for event in events)
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
    result = await zmq_backend.generate(_request("s2", "stoppable text", max_tokens=8, stop=stop))
    completion = result.completions[0]
    assert completion.finish_reason == "stop"
    assert stop not in completion.text


async def test_stop_token_text_omission_crosses_delta_wire(zmq_backend):
    probe = await zmq_backend.generate(
        _request("token-probe", "stoppable token", max_tokens=8)
    )
    stop_id = probe.completions[0].token_ids[1]

    result = await zmq_backend.generate(
        _request(
            "token-stop",
            "stoppable token",
            max_tokens=8,
            stop_token_ids=(stop_id,),
        )
    )
    completion = result.completions[0]

    assert completion.finish_reason == "stop"
    assert completion.token_ids == probe.completions[0].token_ids[:2]
    assert completion.text == f"tok{completion.token_ids[0]}"
    assert f"tok{stop_id}" not in completion.text
    assert result.usage is not None
    assert result.usage.completion_tokens == 2


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


async def test_tensor_parallel_process_service_is_non_daemon_and_attested():
    backend = create_backend(
        "kairyu-proc",
        num_pages=64,
        tensor_parallel_size=2,
    )
    try:
        result = await backend.generate(
            _request("proc-tp2", "process split tensor parallel", max_tokens=2)
        )
        process = backend._process
        assert result.finished is True
        assert backend.tensor_parallel_size == 2
        assert backend._config["tensor_parallel_size"] == 2
        assert process is not None and process.is_alive()
        assert process.daemon is False

        app = create_app(engines={"proc-tp2": backend})
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            payload = (await client.get("/backends")).json()
        assert payload["engines"][0]["engine_backend"] == "kairyu-proc"
        assert payload["engines"][0]["tensor_parallel_size"] == 2
    finally:
        await backend.shutdown()


@pytest.mark.parametrize("value", [0, -1, True, 2.0, "2"])
async def test_tensor_parallel_startup_metadata_rejects_invalid_values(value):
    frame = {
        "startup_wire_version": STARTUP_WIRE_VERSION,
        "status": "ready",
        "tensor_parallel_size": value,
    }
    with pytest.raises(EngineServiceError, match="positive integer"):
        _tensor_parallel_size_from_startup_frame(frame)


async def test_tensor_parallel_startup_identity_is_fail_closed():
    _validate_startup_tensor_parallel_identity(1, None)
    _validate_startup_tensor_parallel_identity(4, 4)
    with pytest.raises(EngineServiceError, match="did not report"):
        _validate_startup_tensor_parallel_identity(4, None)
    with pytest.raises(EngineServiceError, match="mismatch"):
        _validate_startup_tensor_parallel_identity(4, 2)


async def test_engine_service_close_always_shuts_down_parallel_launcher():
    calls = []

    class Launcher:
        def shutdown(self):
            calls.append("launcher")

    class Loop:
        parallel_launcher = Launcher()

        def close(self):
            calls.append("loop")
            raise RuntimeError("loop close failed")

    with pytest.raises(RuntimeError, match="loop close failed"):
        _close_engine_service_loop(Loop())
    assert calls == ["loop", "launcher"]


async def test_engine_service_packs_off_thread_while_depth_one_next_step_runs(
    monkeypatch,
):
    class _OverlapRunner:
        def __init__(self) -> None:
            self.threads: list[int] = []
            self.next_decode_finished = threading.Event()

        def execute(self, scheduled, states):
            self.threads.append(threading.get_ident())
            if any(
                not chunk.is_prefill and chunk.position >= 1
                for chunk in scheduled
            ):
                if not pack_entered.wait(5):
                    raise TimeoutError("wire packing did not start")
                self.next_decode_finished.set()
            return {
                chunk.request_id: tuple(
                    SampledToken(chunk.position + offset)
                    for offset in range(
                        1 if chunk.is_prefill else chunk.num_tokens
                    )
                )
                for chunk in scheduled
                if not chunk.is_prefill
                or states[chunk.request_id].prefill_done
            }

        def release(self, _request_id):
            return None

    runner = _OverlapRunner()
    engine_loop, _, _ = build_engine_loop(
        num_pages=64,
        tokenizer=ToyTokenizer(),
        runner=runner,
        pipeline_depth=1,
    )
    service_socket = _QueuedServiceSocket()
    context = _install_fake_service_zmq(monkeypatch, service_socket)
    port_pipe = _FakeServicePortPipe()
    monkeypatch.setattr(
        "kairyu.engine.kairyu_backend.build_engine_loop",
        lambda **_config: (engine_loop, None, None),
    )

    pack_entered = threading.Event()
    pack_threads: list[int] = []
    pack_gate_lock = threading.Lock()
    pack_gated = False
    real_packb = msgpack.packb
    fake_msgpack = types.ModuleType("msgpack")
    fake_msgpack.unpackb = msgpack.unpackb

    def tracked_packb(payload):
        nonlocal pack_gated
        gate = False
        if (
            isinstance(payload, dict)
            and payload.get("request_id") == "wire-overlap"
            and payload.get("outputs")
        ):
            with pack_gate_lock:
                if not pack_gated:
                    pack_gated = True
                    gate = True
        if gate:
            pack_threads.append(threading.get_ident())
            pack_entered.set()
            if not runner.next_decode_finished.wait(5):
                raise TimeoutError("next depth-one step did not finish")
        return real_packb(payload)

    fake_msgpack.packb = tracked_packb
    monkeypatch.setitem(sys.modules, "msgpack", fake_msgpack)
    service_socket.shutdown_on_terminal = True
    service_socket.enqueue(
        {
            "op": "add",
            "request_id": "wire-overlap",
            "prompt": "one two",
            "sampling": {"max_tokens": 3, "ignore_eos": True},
            "wire_version": LEGACY_WIRE_VERSION,
        }
    )
    run_engine_service(port_pipe, {})

    events = [
        msgpack.unpackb(frames[1])
        for frames in service_socket.sent
        if msgpack.unpackb(frames[1]).get("request_id") == "wire-overlap"
    ]
    output_lengths = [len(event["outputs"]) for event in events]
    assert pack_entered.is_set()
    assert runner.next_decode_finished.is_set()
    assert output_lengths == [1, 2, 3]
    assert [event["finished"] for event in events] == [False, False, True]
    assert pack_threads
    assert set(pack_threads).isdisjoint(runner.threads)
    assert set(pack_threads).isdisjoint(service_socket.send_threads)
    assert set(runner.threads).isdisjoint(service_socket.send_threads)
    assert port_pipe.frames[0] == 43124
    assert port_pipe.frames[1]["status"] == "ready"
    assert service_socket.closed is True
    assert context.terminated is True


async def test_engine_service_inactive_abort_does_not_block_later_socket_ops(
    monkeypatch,
):
    class _LateAbortSocket(_QueuedServiceSocket):
        def __init__(self) -> None:
            super().__init__()
            self._injected = False
            self._staged_polls = 0

        def poll(self, timeout):
            if self._staged is not None:
                self._staged_polls += 1
                if self._staged_polls > 5:
                    raise AssertionError("service stopped draining staged messages")
            return super().poll(timeout)

        def recv_multipart(self):
            self._staged_polls = 0
            return super().recv_multipart()

        def send_multipart(self, frames):
            super().send_multipart(frames)
            payload = msgpack.unpackb(frames[1])
            if (
                not self._injected
                and payload.get("request_id") == "late-abort"
                and payload.get("finished")
            ):
                self._injected = True
                self.enqueue(
                    {"op": "abort", "request_id": "late-abort"},
                )
                self.enqueue({"op": "ping"}, identity=b"probe")
                self.enqueue({"op": "shutdown"}, identity=b"shutdown")

    engine_loop, _, _ = build_engine_loop(
        num_pages=64,
        tokenizer=ToyTokenizer(),
        pipeline_depth=1,
    )
    service_socket = _LateAbortSocket()
    context = _install_fake_service_zmq(monkeypatch, service_socket)
    port_pipe = _FakeServicePortPipe()
    monkeypatch.setattr(
        "kairyu.engine.kairyu_backend.build_engine_loop",
        lambda **_config: (engine_loop, None, None),
    )
    service_socket.enqueue(
        {
            "op": "add",
            "request_id": "late-abort",
            "prompt": "one two",
            "sampling": {"max_tokens": 1, "ignore_eos": True},
            "wire_version": LEGACY_WIRE_VERSION,
        }
    )

    run_engine_service(port_pipe, {})

    replies = [
        (frames[0], msgpack.unpackb(frames[1]))
        for frames in service_socket.sent
    ]
    assert service_socket.terminal_sent.is_set()
    assert (b"probe", {"op": "pong"}) in replies
    assert (b"shutdown", {"op": "bye"}) in replies
    assert service_socket.closed is True
    assert context.terminated is True


@pytest.mark.parametrize(
    ("reported_failure_type", "reported_dead_ranks", "expected_frame"),
    [
        (
            "CollectiveTimeout",
            (),
            {
                "op": "fatal",
                "error_type": "CollectiveTimeout",
                "dead_ranks": [],
            },
        ),
        (
            None,
            (1,),
            {
                "op": "fatal",
                "error_type": "RankProcessExited",
                "dead_ranks": [1],
            },
        ),
    ],
)
async def test_engine_service_ping_reports_parallel_failure_and_cleans_up(
    monkeypatch,
    reported_failure_type,
    reported_dead_ranks,
    expected_frame,
):
    class Launcher:
        def __init__(self) -> None:
            self.failure_checks = 0
            self.rank_checks = 0
            self.shutdown_calls = 0

        def failure_type(self):
            self.failure_checks += 1
            if self.failure_checks == 1:
                return None
            return reported_failure_type

        def dead_ranks(self):
            self.rank_checks += 1
            if self.rank_checks == 1:
                return ()
            return reported_dead_ranks

        def shutdown(self):
            self.shutdown_calls += 1

    class Loop:
        tensor_parallel_size = 2
        generation_defaults = None

        def __init__(self) -> None:
            self.parallel_launcher = Launcher()
            self.closed = False

        def has_work(self):
            return False

        def close(self):
            self.closed = True

    loop = Loop()
    port_pipe = _FakeServicePortPipe()
    service_socket = _FakeServiceSocket({"op": "ping"})
    context = _install_fake_service_zmq(monkeypatch, service_socket)
    monkeypatch.setattr(
        "kairyu.engine.kairyu_backend.build_engine_loop",
        lambda **_config: (loop, None, None),
    )

    run_engine_service(port_pipe, {"tensor_parallel_size": 2})

    assert port_pipe.frames[0] == 43123
    assert port_pipe.frames[1]["status"] == "ready"
    assert port_pipe.closed is True
    assert len(service_socket.sent) == 1
    assert msgpack.unpackb(service_socket.sent[0][1]) == expected_frame
    assert loop.closed is True
    assert loop.parallel_launcher.shutdown_calls == 1
    assert service_socket.closed is True
    assert context.terminated is True


async def test_startup_rank_attestation_failure_cleans_every_service_resource(
    monkeypatch,
):
    class Launcher:
        def __init__(self) -> None:
            self.shutdown_calls = 0

        def failure_type(self):
            return None

        def dead_ranks(self):
            return (1,)

        def shutdown(self):
            self.shutdown_calls += 1

    class Loop:
        tensor_parallel_size = 2
        generation_defaults = None

        def __init__(self) -> None:
            self.parallel_launcher = Launcher()
            self.closed = False

        def close(self):
            self.closed = True

    loop = Loop()
    port_pipe = _FakeServicePortPipe()
    service_socket = _FakeServiceSocket()
    context = _install_fake_service_zmq(monkeypatch, service_socket)
    monkeypatch.setattr(
        "kairyu.engine.kairyu_backend.build_engine_loop",
        lambda **_config: (loop, None, None),
    )

    with pytest.raises(RuntimeError, match="failed before startup readiness"):
        run_engine_service(port_pipe, {"tensor_parallel_size": 2})

    assert port_pipe.frames[0] == 43123
    assert port_pipe.frames[1]["status"] == "error"
    assert port_pipe.frames[1]["error_type"] == "RuntimeError"
    assert port_pipe.closed is True
    assert loop.closed is True
    assert loop.parallel_launcher.shutdown_calls == 1
    assert service_socket.closed is True
    assert context.terminated is True


async def test_forced_cleanup_kills_nested_rank_process_group(monkeypatch):
    monkeypatch.setattr(
        "kairyu.engine.zmq_backend.run_engine_service",
        _fake_nested_engine_service,
    )
    spawn = multiprocessing.get_context("spawn")
    rank_pid_recv, rank_pid_send = spawn.Pipe(duplex=False)
    backend = ZmqEngineBackend(num_pages=64, tensor_parallel_size=2)
    backend._config["_test_rank_pid_pipe"] = rank_pid_send
    process = None
    rank_pid = None
    try:
        assert await asyncio.to_thread(backend._spawn) == 43216
        rank_pid_send.close()
        process = backend._process
        assert await asyncio.to_thread(rank_pid_recv.poll, 5)
        rank_pid = rank_pid_recv.recv()
        assert process is not None and process.is_alive()
        assert isinstance(rank_pid, int) and Path(f"/proc/{rank_pid}").exists()

        assert await asyncio.to_thread(_kill_and_reap_process, process, 3.0)
        assert not Path(f"/proc/{rank_pid}").exists()
        assert _process_group_member_states(process) == ()
    finally:
        if process is not None:
            _kill_and_reap_process(process, 1.0)
        backend._close_transport_locked()
        backend._process = None
        rank_pid_recv.close()
        rank_pid_send.close()


async def test_parent_death_lease_kills_service_and_nested_rank():
    _ensure_child_subreaper()
    spawn = multiprocessing.get_context("spawn")
    result_recv, result_send = spawn.Pipe(duplex=False)
    api_parent = spawn.Process(
        target=_fake_api_parent_with_nested_service,
        args=(result_send,),
    )
    api_parent.start()
    result_send.close()
    service_pid = rank_pid = None
    try:
        assert await asyncio.to_thread(result_recv.poll, 10)
        service_pid, rank_pid = result_recv.recv()
        assert all(
            Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
            for pid in (service_pid, rank_pid)
        )
        os.kill(api_parent.pid, signal.SIGKILL)
        await asyncio.to_thread(api_parent.join, 3)
        assert not api_parent.is_alive()
        for _ in range(300):
            present = []
            for pid in (service_pid, rank_pid):
                try:
                    state = Path(f"/proc/{pid}/stat").read_text().split()[2]
                except (FileNotFoundError, ProcessLookupError):
                    continue
                if state == "Z":
                    with contextlib.suppress(ChildProcessError):
                        os.waitpid(pid, os.WNOHANG)
                if Path(f"/proc/{pid}").exists():
                    present.append(pid)
            if not present:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail(f"parent death left unreaped service tree: {present}")
    finally:
        if api_parent.is_alive():
            api_parent.kill()
            api_parent.join(timeout=1)
        if service_pid is not None and os.name == "posix":
            with contextlib.suppress(ProcessLookupError):
                os.killpg(service_pid, signal.SIGKILL)
        result_recv.close()
        result_send.close()


async def test_shutdown_kills_sigterm_resistant_nested_rank(
    monkeypatch,
):
    monkeypatch.setattr(
        "kairyu.engine.zmq_backend.run_engine_service",
        _fake_nested_engine_service,
    )
    monkeypatch.setattr("kairyu.engine.zmq_backend._SHUTDOWN_TIMEOUT_S", 0.05)
    spawn = multiprocessing.get_context("spawn")
    rank_pid_recv, rank_pid_send = spawn.Pipe(duplex=False)
    backend = ZmqEngineBackend(num_pages=64, tensor_parallel_size=2)
    backend._config.update(
        {
            "_test_rank_pid_pipe": rank_pid_send,
            "_test_rank_ignore_sigterm": True,
        }
    )
    rank_pid = None
    try:
        assert await asyncio.to_thread(backend._spawn) == 43216
        rank_pid_send.close()
        assert await asyncio.to_thread(rank_pid_recv.poll, 5)
        rank_pid = rank_pid_recv.recv()
        await backend.shutdown()
        assert not Path(f"/proc/{rank_pid}").exists()
    finally:
        process = backend._process
        if process is not None:
            _kill_and_reap_process(process, 1.0)
        backend._process = None
        rank_pid_recv.close()
        rank_pid_send.close()


async def test_receiver_reaps_followers_when_service_rank_dies(
    monkeypatch,
):
    monkeypatch.setattr(
        "kairyu.engine.zmq_backend.run_engine_service",
        _fake_nested_engine_service,
    )
    monkeypatch.setattr("kairyu.engine.zmq_backend._RECV_TICK_S", 0.01)
    spawn = multiprocessing.get_context("spawn")
    rank_pid_recv, rank_pid_send = spawn.Pipe(duplex=False)
    backend = ZmqEngineBackend(num_pages=64, tensor_parallel_size=2)
    backend._config["_test_rank_pid_pipe"] = rank_pid_send

    class BlockingSocket:
        async def recv(self):
            await asyncio.Future()

    process = None
    rank_pid = None
    try:
        assert await asyncio.to_thread(backend._spawn) == 43216
        rank_pid_send.close()
        process = backend._process
        assert await asyncio.to_thread(rank_pid_recv.poll, 5)
        rank_pid = rank_pid_recv.recv()
        assert process is not None
        backend._live_generation = 1
        receiver = asyncio.create_task(
            backend._receive_loop(1, BlockingSocket(), process)
        )

        # Kill rank 0 only. Its follower deliberately remains in the owned
        # process group until the receiver observes death and sweeps the group.
        os.kill(process.pid, signal.SIGKILL)
        await asyncio.wait_for(receiver, timeout=3)
        assert not process.is_alive()
        assert not Path(f"/proc/{rank_pid}").exists()
        assert _process_group_member_states(process) == ()
        assert backend._process is None
    finally:
        owned_process = backend._process
        if owned_process is not None:
            _kill_and_reap_process(owned_process, 1.0)
        backend._close_transport_locked()
        backend._process = None
        rank_pid_recv.close()
        rank_pid_send.close()


async def test_receiver_transport_failure_marks_readiness_fatal_and_sanitized(
    monkeypatch,
):
    class LiveProcess:
        pid = None

        def is_alive(self):
            return True

    class FailingSocket:
        async def recv(self):
            raise RuntimeError("secret transport path /srv/private")

    signals = []
    monkeypatch.setattr(
        "kairyu.engine.zmq_backend._signal_process_tree",
        lambda _process, sig: signals.append(sig),
    )
    monkeypatch.setattr(
        "kairyu.engine.zmq_backend._wait_process_tree_exit",
        lambda _process, _timeout: True,
    )
    backend = ZmqEngineBackend(num_pages=64)
    process = LiveProcess()
    queue = asyncio.Queue()
    backend._process = process
    backend._live_generation = 3
    backend._queues["failed"] = queue
    backend._queue_generations["failed"] = 3
    receiver = asyncio.create_task(
        backend._receive_loop(3, FailingSocket(), process)
    )
    backend._receiver = receiver

    await asyncio.wait_for(asyncio.shield(receiver), timeout=0.5)

    event = queue.get_nowait()
    assert "secret transport path" in event["error"]
    status = backend.readiness()
    assert status.ready is False
    assert status.fatal is True
    assert status.detail == "engine service failed: RuntimeError"
    assert "private" not in status.detail
    assert signals == [signal.SIGKILL]
    assert backend._process is None
    backend._receiver = None


async def test_receiver_updates_live_decode_graph_metadata_without_dropping_events(
    caplog,
):
    class LiveProcess:
        pid = None

        def is_alive(self):
            return True

    valid = {
        "decode_mode": "cuda_graph",
        "cuda_graph_decode": True,
        "cuda_graph_buckets": [1, 2],
        "cuda_graph_captures": 2,
        "cuda_graph_replays": 4,
        "cuda_graph_eager_fallbacks": 1,
    }
    malformed = {**valid, "cuda_graph_eager_fallbacks": -1}

    class Socket:
        def __init__(self):
            self.messages = [
                msgpack.packb(
                    {
                        "request_id": "wire",
                        "finished": False,
                        "decode_graph_metadata": valid,
                    }
                ),
                msgpack.packb(
                    {
                        "request_id": "wire",
                        "finished": True,
                        "decode_graph_metadata": malformed,
                    }
                ),
            ]

        async def recv(self):
            if self.messages:
                return self.messages.pop(0)
            await asyncio.Future()

    backend = ZmqEngineBackend(num_pages=64)
    backend._live_generation = 23
    backend._public_request_ids["wire"] = "public"
    backend._queues["public"] = asyncio.Queue()
    receiver = asyncio.create_task(
        backend._receive_loop(23, Socket(), LiveProcess())
    )
    try:
        first = await asyncio.wait_for(backend._queues["public"].get(), 0.5)
        second = await asyncio.wait_for(backend._queues["public"].get(), 0.5)
        assert first["finished"] is False
        assert second["finished"] is True
        assert backend.decode_graph_metadata_snapshot() == {
            **valid,
            "cuda_graph_buckets": (1, 2),
        }
        assert "ignored malformed decode-graph metadata" in caplog.text
    finally:
        receiver.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await receiver


async def test_receiver_heartbeat_timeout_fails_generation_and_readiness(
    monkeypatch,
):
    class LiveProcess:
        pid = None

        def is_alive(self):
            return True

    class SilentSocket:
        def __init__(self) -> None:
            self.sent = []

        async def recv(self):
            await asyncio.Future()

        async def send(self, payload):
            self.sent.append(msgpack.unpackb(payload))

    signals = []
    monkeypatch.setattr("kairyu.engine.zmq_backend._RECV_TICK_S", 0.005)
    monkeypatch.setattr(
        "kairyu.engine.zmq_backend._signal_process_tree",
        lambda _process, sig: signals.append(sig),
    )
    monkeypatch.setattr(
        "kairyu.engine.zmq_backend._wait_process_tree_exit",
        lambda _process, _timeout: True,
    )
    backend = ZmqEngineBackend(num_pages=64, death_timeout_s=0.03)
    process = LiveProcess()
    socket = SilentSocket()
    queue = asyncio.Queue()
    backend._process = process
    backend._live_generation = 5
    backend._queues["silent"] = queue
    backend._queue_generations["silent"] = 5
    receiver = asyncio.create_task(backend._receive_loop(5, socket, process))
    backend._receiver = receiver
    try:
        await asyncio.wait_for(asyncio.shield(receiver), timeout=0.5)
    finally:
        if not receiver.done():
            receiver.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await receiver

    assert socket.sent and all(message == {"op": "ping"} for message in socket.sent)
    assert "heartbeat timed out" in queue.get_nowait()["error"]
    status = backend.readiness()
    assert status.ready is False
    assert status.fatal is True
    assert status.detail == "engine service failed: EngineServiceError"
    assert signals == [signal.SIGKILL]
    assert backend._process is None
    backend._receiver = None


async def test_receiver_retires_generation_before_blocked_reap(monkeypatch):
    class LiveProcess:
        pid = None

        def is_alive(self):
            return True

    class FailingSocket:
        async def recv(self):
            raise RuntimeError("transport failed")

    reap_entered = threading.Event()
    release_reap = threading.Event()
    reap_calls = []

    def controlled_reap(process, timeout_s):
        reap_calls.append((process, timeout_s))
        reap_entered.set()
        assert release_reap.wait(timeout=2)
        return True

    monkeypatch.setattr(
        "kairyu.engine.zmq_backend._signal_process_tree",
        lambda _process, _sig: None,
    )
    monkeypatch.setattr(
        "kairyu.engine.zmq_backend._wait_process_tree_exit",
        controlled_reap,
    )
    backend = ZmqEngineBackend(num_pages=64)
    process = LiveProcess()
    queue = asyncio.Queue()
    backend._process = process
    backend._live_generation = 17
    backend._queues["blocked-reap"] = queue
    backend._queue_generations["blocked-reap"] = 17
    receiver = asyncio.create_task(
        backend._receive_loop(17, FailingSocket(), process)
    )
    backend._receiver = receiver
    try:
        assert await asyncio.to_thread(reap_entered.wait, 1)

        # Reaping is still blocked and the fake process is still alive, yet
        # admission and readiness must already expose the fatal generation.
        assert receiver.done() is False
        assert backend._live_generation is None
        assert backend._is_healthy() is False
        status = backend.readiness()
        assert status.ready is False and status.fatal is True
        assert status.detail == "engine service failed: RuntimeError"
        assert "transport failed" in queue.get_nowait()["error"]
    finally:
        release_reap.set()
        await asyncio.wait_for(asyncio.shield(receiver), timeout=1)

    assert reap_calls == [(process, 5.0)]
    assert backend._process is None
    backend._receiver = None


async def test_cancelled_reset_shares_and_drains_receiver_reap(monkeypatch):
    class LiveProcess:
        pid = None

        def is_alive(self):
            return True

    class FailingSocket:
        async def recv(self):
            raise RuntimeError("transport failed")

    reap_entered = threading.Event()
    release_reap = threading.Event()
    reap_calls = []
    signals = []

    def controlled_reap(process, timeout_s):
        reap_calls.append((process, timeout_s))
        reap_entered.set()
        assert release_reap.wait(timeout=2)
        return True

    monkeypatch.setattr(
        "kairyu.engine.zmq_backend._signal_process_tree",
        lambda _process, sig: signals.append(sig),
    )
    monkeypatch.setattr(
        "kairyu.engine.zmq_backend._wait_process_tree_exit",
        controlled_reap,
    )
    backend = ZmqEngineBackend(num_pages=64)
    process = LiveProcess()
    backend._process = process
    backend._live_generation = 19
    receiver = asyncio.create_task(
        backend._receive_loop(19, FailingSocket(), process)
    )
    backend._receiver = receiver
    reset = None
    try:
        assert await asyncio.to_thread(reap_entered.wait, 1)
        reset = asyncio.create_task(backend._reset_dead_locked())
        for _ in range(100):
            if backend._receiver is None:
                break
            await asyncio.sleep(0)
        assert backend._receiver is None

        reset.cancel()
        await asyncio.sleep(0)
        assert reset.done() is False
        release_reap.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(reset, timeout=1)
    finally:
        release_reap.set()
        if reset is not None and not reset.done():
            with contextlib.suppress(asyncio.CancelledError):
                await reset
        if not receiver.done():
            receiver.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await receiver

    assert reap_calls == [(process, 5.0)]
    assert signals == [signal.SIGKILL]
    assert receiver.done()
    assert backend._process is None


async def test_rejects_non_string_tokenizer():
    with pytest.raises(ValueError, match="string tokenizer"):
        ZmqEngineBackend(tokenizer=ToyTokenizer())  # type: ignore[arg-type]


async def test_process_backend_resolves_loglikelihood_boundary_without_starting_child():
    backend = ZmqEngineBackend(tokenizer="toy")

    context_ids, continuation_ids = backend.tokenize_loglikelihood(
        "Question Answer:", " A"
    )

    tokenizer = ToyTokenizer()
    assert context_ids == tokenizer.encode("Question Answer:")
    assert context_ids + continuation_ids == tokenizer.encode("Question Answer: A")
    assert backend._process is None


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), "10"])
async def test_rejects_invalid_death_timeout(value):
    with pytest.raises(ValueError, match="death_timeout_s"):
        ZmqEngineBackend(death_timeout_s=value)


async def test_model_less_constructor_rejects_unshardable_tp_degree():
    with pytest.raises(ValueError, match="does not divide"):
        ZmqEngineBackend(tensor_parallel_size=3)


@pytest.mark.parametrize(
    ("sampling", "tools", "field"),
    [
        ({"best_of": 2}, (), "best_of"),
        ({"prompt_logprobs": 1}, (), "prompt_logprobs"),
        ({"extra_args": {"unsupported": True}}, (), "extra_args.unsupported"),
    ],
)
async def test_native_process_layouts_reject_the_same_unsupported_surface(
    sampling,
    tools,
    field,
):
    request = GenerationRequest(
        request_id=f"unsupported-{field}",
        prompt="surface parity",
        sampling_params=SamplingParams(**sampling),
        tools=tools,
    )
    in_process = KairyuBackend(num_pages=64)
    process_split = ZmqEngineBackend(num_pages=64)
    try:
        errors = []
        for backend in (in_process, process_split):
            with pytest.raises(ValueError) as caught:
                backend.validate_request(request)
            errors.append(str(caught.value))

        assert errors[0] == errors[1]
        assert field in errors[0]
        assert process_split._process is None
    finally:
        await in_process.shutdown()
        await process_split.shutdown()


async def test_process_wire_carries_native_strict_tool_structural_tag():
    request = GenerationRequest(
        "strict-wire",
        "prompt",
        SamplingParams(),
        tools=(
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
        ),
        tool_call_protocol="qwen",
    )

    packed = ZmqEngineBackend._pack_add_message(
        request,
        None,
        GenerationDefaults(),
        "wire-strict",
        "stream-strict",
        WIRE_VERSION,
    )
    message = msgpack.unpackb(packed)
    response_format = message["sampling"]["extra_args"]["response_format"]

    assert response_format["type"] == "structural_tag"
    assert response_format["format"]["tags"][0]["content"]["style"] == (
        "qwen_xml"
    )


async def test_derived_full_validator_is_not_bypassed_by_builtin_fast_hook():
    class RejectingDerivedBackend(ZmqEngineBackend):
        def __init__(self):
            super().__init__(num_pages=64)
            self.validation_calls = 0

        def validate_request(self, request):
            self.validation_calls += 1
            super().validate_request(request)
            raise ValueError("derived rejection")

    backend = RejectingDerivedBackend()
    try:
        with pytest.raises(ValueError, match="derived rejection"):
            await backend.prepare_request(
                _request("derived-validator", "hello", max_tokens=1)
            )
        assert backend.validation_calls == 1
        assert backend._process is None
    finally:
        await backend.shutdown()


async def test_native_process_layouts_accept_response_format_extension():
    request = _request(
        "response-format-parity",
        "surface parity",
        max_tokens=1,
        extra_args={"response_format": {"type": "json_object"}},
    )
    tokenizer = ToyTokenizer()
    tokenizer.eos_token_id = 0
    in_process = KairyuBackend(
        num_pages=64,
        tokenizer=tokenizer,
        max_model_len=32,
    )
    process_split = ZmqEngineBackend(num_pages=64, max_model_len=32)
    process_split._preflight_tokenizer = tokenizer
    try:
        in_process.validate_request(request)
        process_split.validate_request(request)
    finally:
        await in_process.shutdown()
        await process_split.shutdown()


async def test_process_sync_validation_rejects_malformed_grammar_without_context_limit():
    tokenizer = ToyTokenizer()
    tokenizer.eos_token_id = 0
    backend = ZmqEngineBackend(num_pages=64)
    backend._preflight_tokenizer = tokenizer
    request = _request(
        "malformed-grammar",
        "surface parity",
        max_tokens=1,
        extra_args={"response_format": {"type": "regex", "pattern": "("}},
    )

    try:
        with pytest.raises(ValueError, match="invalid structured output"):
            backend.validate_request(request)

        assert backend._process is None
    finally:
        await backend.shutdown()


async def test_process_structured_validation_caches_fresh_custom_vocabulary():
    class FreshVocabularyTokenizer(ToyTokenizer):
        def __init__(self) -> None:
            self.eos_token_id = 0
            self.vocab_calls = 0

        def vocab(self) -> list[str]:
            self.vocab_calls += 1
            return super().vocab()

    tokenizer = FreshVocabularyTokenizer()
    backend = ZmqEngineBackend(num_pages=64)
    backend._preflight_tokenizer = tokenizer
    try:
        for request_id, pattern in (("first", "a+"), ("second", "b+")):
            backend.validate_request(
                _request(
                    request_id,
                    "surface parity",
                    max_tokens=1,
                    extra_args={
                        "response_format": {
                            "type": "regex",
                            "pattern": pattern,
                        }
                    },
                )
            )

        assert tokenizer.vocab_calls == 1
    finally:
        await backend.shutdown()


async def test_native_sync_validation_rejects_unserializable_tool_schema():
    request = GenerationRequest(
        request_id="unserializable-tool",
        prompt="surface parity",
        sampling_params=SamplingParams(max_tokens=1),
        tools=(
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"invalid": object()},
                },
            },
        ),
        tool_choice="required",
    )
    in_process = KairyuBackend(num_pages=64)
    process_split = ZmqEngineBackend(num_pages=64)
    try:
        for backend in (in_process, process_split):
            with pytest.raises(TypeError, match="JSON serializable"):
                backend.validate_request(request)
        assert process_split._process is None
    finally:
        await in_process.shutdown()
        await process_split.shutdown()


async def test_native_process_layouts_accept_explicit_special_token_output_policy():
    request = _request(
        "special-policy-parity",
        "surface parity",
        max_tokens=1,
        skip_special_tokens=False,
    )
    in_process = KairyuBackend(num_pages=64)
    process_split = ZmqEngineBackend(num_pages=64)
    try:
        in_process.validate_request(request)
        process_split.validate_request(request)
        assert process_split._process is None
    finally:
        await in_process.shutdown()
        await process_split.shutdown()


async def test_cuda_graph_serving_options_cross_the_process_boundary():
    backend = ZmqEngineBackend(
        model_path="/models/tiny",
        max_model_len=8192,
        pipeline_depth=2,
        decode_mode="cuda_graph",
        cuda_graph_max_batch=4,
        cuda_graph_max_pages=32,
        cuda_graph_warmup_iters=1,
    )

    assert backend._config["max_model_len"] == 8192
    assert backend._config["pipeline_depth"] == 2
    assert backend._config["decode_mode"] == "cuda_graph"
    assert backend._config["cuda_graph_max_batch"] == 4
    assert backend._config["cuda_graph_max_pages"] == 32
    assert backend._config["cuda_graph_warmup_iters"] == 1


async def test_max_model_len_8192_reaches_the_spawned_child(monkeypatch):
    monkeypatch.setattr(
        "kairyu.engine.zmq_backend.run_engine_service",
        _fake_max_model_len_engine_service,
    )
    backend = ZmqEngineBackend(num_pages=64, max_model_len=8192)
    try:
        port = await asyncio.to_thread(backend._spawn)

        assert port == 43214
        assert backend.attention_backend_decision is None
    finally:
        await backend._reset_dead_locked()


async def test_parent_preflight_enforces_context_limit_without_starting_child():
    backend = ZmqEngineBackend(num_pages=64, max_model_len=5)
    oversized = _request(
        "context-retry",
        TokensPrompt((1, 2, 3, 4)),
        max_tokens=2,
    )

    try:
        with pytest.raises(ValueError, match="exceed max_model_len"):
            backend.validate_request(oversized)
        with pytest.raises(ValueError, match="exceed max_model_len"):
            await backend.prepare_request(oversized)

        assert backend._process is None
        assert backend._active_request_ids == set()
        assert backend._queues == {}
        assert backend._prepared_requests == {}

        boundary = await backend.generate(
            _request(
                "context-retry",
                TokensPrompt((1, 2, 3)),
                max_tokens=2,
            )
        )
        assert boundary.finished
        assert boundary.usage is not None
        assert boundary.usage.prompt_tokens + boundary.usage.completion_tokens == 5
    finally:
        await backend.shutdown()


async def test_abandoned_parent_preflight_does_not_retain_request():
    backend = ZmqEngineBackend(num_pages=64, max_model_len=32)
    request = _request("abandoned-preflight", "fits", max_tokens=1)
    await backend.prepare_request(request)
    request_ref = weakref.ref(request)

    assert len(backend._prepared_requests) == 1
    del request
    gc.collect()

    assert request_ref() is None
    assert backend._prepared_requests == {}
    await backend.shutdown()


async def test_concurrent_parent_prepare_single_flights_same_request():
    tokenizer = _GatedTokenizer()
    backend = ZmqEngineBackend(num_pages=64, max_model_len=32)
    backend._preflight_tokenizer = tokenizer
    request = _request("parent-single-flight", "shared request", max_tokens=1)
    first = asyncio.create_task(backend.prepare_request(request))
    second = asyncio.create_task(backend.prepare_request(request))
    try:
        assert await asyncio.to_thread(tokenizer.entered.wait, 1)
        for _ in range(100):
            flight = backend._preparing_requests.get(id(request))
            if flight is not None and flight.waiters == 2:
                break
            await asyncio.sleep(0)
        else:
            pytest.fail("same-request parent preflight did not join one flight")

        tokenizer.release.set()
        await asyncio.gather(first, second)

        assert tokenizer.calls == 1
        assert backend._peek_prepared_request(request) is not None
        assert backend._preparing_requests == {}
        assert backend._process is None
    finally:
        tokenizer.release.set()
        await asyncio.gather(first, second, return_exceptions=True)
        await backend.shutdown()


async def test_different_parent_requests_serialize_custom_tokenizer_calls():
    tokenizer = _OverlapRejectingTokenizer()
    backend = ZmqEngineBackend(num_pages=64, max_model_len=32)
    backend._preflight_tokenizer = tokenizer
    first = asyncio.create_task(
        backend.prepare_request(
            _request("serialize-parent-a", "first", max_tokens=1)
        )
    )
    second = asyncio.create_task(
        backend.prepare_request(
            _request("serialize-parent-b", "second", max_tokens=1)
        )
    )
    try:
        assert await asyncio.to_thread(tokenizer.entered.wait, 1)
        await asyncio.sleep(0.05)
        tokenizer.release.set()
        await asyncio.gather(first, second)

        assert tokenizer.calls == 2
        assert backend._process is None
    finally:
        tokenizer.release.set()
        await asyncio.gather(first, second, return_exceptions=True)
        await backend.shutdown()


async def test_parent_loglikelihood_shares_custom_tokenizer_gate_with_prepare():
    tokenizer = _OverlapRejectingTokenizer()
    backend = ZmqEngineBackend(num_pages=64, max_model_len=32)
    backend._preflight_tokenizer = tokenizer
    preparation = asyncio.create_task(
        backend.prepare_request(
            _request("serialize-parent-prepare", "prompt", max_tokens=1)
        )
    )
    scoring = None
    try:
        assert await asyncio.to_thread(tokenizer.entered.wait, 1)
        scoring = asyncio.create_task(
            backend.tokenize_loglikelihood_async("context", " continuation")
        )
        await asyncio.sleep(0.05)
        assert tokenizer.calls == 1

        tokenizer.release.set()
        await asyncio.gather(preparation, scoring)
        assert tokenizer.calls == 3
        assert backend._process is None
    finally:
        tokenizer.release.set()
        pending = [preparation]
        if scoring is not None:
            pending.append(scoring)
        await asyncio.gather(*pending, return_exceptions=True)
        await backend.shutdown()


async def test_cancelled_parent_prepare_worker_does_not_publish_cache():
    tokenizer = _GatedTokenizer()
    backend = ZmqEngineBackend(num_pages=64, max_model_len=32)
    backend._preflight_tokenizer = tokenizer
    request = _request("cancelled-parent-prepare", "abandoned", max_tokens=1)
    preparation = asyncio.create_task(backend.prepare_request(request))
    try:
        assert await asyncio.to_thread(tokenizer.entered.wait, 1)
        preparation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await preparation
        assert backend._prepared_requests == {}

        tokenizer.release.set()
        for _ in range(100):
            if not backend._preparing_requests:
                break
            await asyncio.sleep(0.01)

        assert tokenizer.calls == 1
        assert backend._preparing_requests == {}
        assert backend._prepared_requests == {}
        assert backend._process is None
    finally:
        tokenizer.release.set()
        await asyncio.gather(preparation, return_exceptions=True)
        await backend.shutdown()


@pytest.mark.parametrize(
    "wire_prompt",
    [
        "one two",
        [1, 2],
    ],
)
async def test_streaming_context_rejection_is_http_400_before_sse_headers(
    wire_prompt,
):
    backend = ZmqEngineBackend(num_pages=64, max_model_len=3)
    app = create_app(engines={"limited-proc": backend})
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/completions",
                json={
                    "model": "limited-proc",
                    "prompt": wire_prompt,
                    "max_tokens": 2,
                    "stream": True,
                },
            )

        assert response.status_code == 400
        assert "exceed max_model_len" in response.json()["error"]["message"]
        assert not response.headers["content-type"].startswith("text/event-stream")
        assert backend._process is None
        assert backend._prepared_requests == {}
    finally:
        await backend.shutdown()


async def test_parent_preflight_tokenizes_tool_prompt_once_and_submits_tokens(
    monkeypatch,
):
    import msgpack

    event_loop_thread = threading.get_ident()

    class CapturingSocket:
        payload = None

        async def send(self, payload):
            self.payload = payload

    tokenizer = _CountingTokenizer()
    monkeypatch.setattr(
        "kairyu.engine.zmq_backend.resolve_tokenizer",
        lambda _source: tokenizer,
    )
    backend = ZmqEngineBackend(
        num_pages=64,
        tokenizer="counting",
        max_model_len=512,
    )
    request = GenerationRequest(
        request_id="prepared-tool",
        prompt=TextPrompt("use a tool"),
        sampling_params=SamplingParams(max_tokens=2),
        trace_requested=True,
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
    socket = CapturingSocket()
    clock = iter((100, 175))
    monkeypatch.setattr(
        "kairyu.engine.zmq_backend.time.perf_counter_ns",
        lambda: next(clock),
    )

    async def already_started():
        return None

    monkeypatch.setattr(backend, "_ensure_started", already_started)
    backend._socket = socket
    backend._live_generation = 1

    await backend.prepare_request(request)
    await backend.prepare_request(request)
    queue = await backend._submit(request)

    assert socket.payload is not None
    message = msgpack.unpackb(socket.payload)
    assert message["prompt"]["kind"] == "tokens"
    assert len(tokenizer.encoded) == 1
    assert tokenizer.encode_threads != [event_loop_thread]
    assert "Available functions:" in tokenizer.encoded[0]
    assert "You must call one of the available functions." in tokenizer.encoded[0]
    assert message["prompt"]["prompt"] == tokenizer.encoded[0]
    assert tuple(message["prompt"]["prompt_token_ids"]) == ToyTokenizer().encode(
        tokenizer.encoded[0]
    )
    assert message["trace_requested"] is True
    assert backend._parent_tokenize_durations_ns[request.request_id] == 75
    result = backend._result(
        request,
        {
            "text": "",
            "outputs": (),
            "finished": True,
            "stage_metrics": (GenerationStageMetric("tokenize", 25),),
        },
    )
    tokenize = result.stage_metrics[0]
    assert tokenize.stage == "tokenize"
    assert tokenize.duration_ns == 100
    assert tokenize.occurrences == 2
    assert backend._prepared_requests == {}
    assert backend._queues.pop(request.request_id) is queue
    backend._release_wire_route(request.request_id)
    assert request.request_id not in backend._parent_tokenize_durations_ns
    backend._socket = None
    backend._live_generation = None
    await backend.shutdown()


async def test_templated_text_is_parent_tokenized_before_versioned_prompt_wire(
    monkeypatch,
):
    import msgpack

    class CapturingSocket:
        payload = None

        async def send(self, payload):
            self.payload = payload

    backend = ZmqEngineBackend(num_pages=64)
    backend.generation_defaults = GenerationDefaults()
    tokenizer = _CountingTokenizer()
    backend._preflight_tokenizer = tokenizer
    event_loop_thread = threading.get_ident()
    pack_threads = []
    original_pack = backend._pack_add_message

    def tracked_pack(*args):
        pack_threads.append(threading.get_ident())
        return original_pack(*args)

    monkeypatch.setattr(backend, "_pack_add_message", tracked_pack)
    socket = CapturingSocket()
    backend._socket = socket
    backend._live_generation = 1

    async def already_started():
        return None

    monkeypatch.setattr(backend, "_ensure_started", already_started)
    request = GenerationRequest(
        "templated-wire",
        TemplatedPrompt("<BOS>user hello<ASSISTANT>"),
        SamplingParams(max_tokens=1),
    )

    queue = await backend._submit(request)

    message = msgpack.unpackb(socket.payload)
    assert message["prompt_wire_version"] == 1
    assert message["prompt"] == {
        "kind": "tokens",
        "prompt_token_ids": list(
            ToyTokenizer().encode("<BOS>user hello<ASSISTANT>")
        ),
        "prompt": "<BOS>user hello<ASSISTANT>",
    }
    assert tokenizer.encoded == ["<BOS>user hello<ASSISTANT>"]
    assert tokenizer.encode_threads != [event_loop_thread]
    assert pack_threads and pack_threads != [event_loop_thread]
    assert backend._queues.pop(request.request_id) is queue
    backend._release_wire_route(request.request_id)
    backend._socket = None
    backend._live_generation = None
    await backend.shutdown()


async def test_submit_negotiates_trace_only_for_opted_in_v2_requests(monkeypatch):
    class CapturingSocket:
        def __init__(self):
            self.messages = []

        async def send(self, payload):
            self.messages.append(msgpack.unpackb(payload))

    backend = ZmqEngineBackend(num_pages=64)
    backend.generation_defaults = GenerationDefaults()
    socket = CapturingSocket()
    backend._socket = socket
    backend._live_generation = 1

    async def already_started():
        return None

    monkeypatch.setattr(backend, "_ensure_started", already_started)
    requests = (
        _request("trace-wire-off", "hello", max_tokens=1),
        _request(
            "trace-wire-on",
            "hello",
            max_tokens=1,
            trace_requested=True,
        ),
    )
    queues = [await backend._submit(request) for request in requests]
    backend._wire_version = LEGACY_WIRE_VERSION
    legacy_request = _request(
        "trace-wire-legacy",
        "hello",
        max_tokens=1,
        trace_requested=True,
    )
    legacy_queue = await backend._submit(legacy_request)

    trace_off, trace_on, legacy = socket.messages
    assert trace_off["wire_version"] == WIRE_VERSION
    assert "trace_requested" not in trace_off
    assert trace_on["wire_version"] == WIRE_VERSION
    assert trace_on["trace_requested"] is True
    assert {"wire_version", "stream_id", "trace_requested"}.isdisjoint(legacy)

    for request, queue in zip(requests, queues, strict=True):
        assert backend._queues.pop(request.request_id) is queue
        backend._release_wire_route(request.request_id)
    assert backend._queues.pop(legacy_request.request_id) is legacy_queue
    backend._release_wire_route(legacy_request.request_id)
    backend._socket = None
    backend._live_generation = None
    await backend.shutdown()


async def test_startup_hook_ensures_the_service_once(monkeypatch):
    backend = ZmqEngineBackend(num_pages=64)
    calls = 0

    async def ensure_started():
        nonlocal calls
        calls += 1

    monkeypatch.setattr(backend, "_ensure_started", ensure_started)

    await backend.startup()

    assert calls == 1


async def test_versioned_startup_frame_propagates_the_child_actual_decision(
    monkeypatch,
):
    monkeypatch.setattr(
        "kairyu.engine.zmq_backend.run_engine_service",
        _fake_ready_engine_service,
    )
    backend = ZmqEngineBackend(num_pages=64)
    try:
        port = await asyncio.to_thread(backend._spawn)
        assert port == 43210
        assert backend.attention_backend_decision == AttentionBackendDecision(
            requested="flashattention4",
            resolved="flashattention4",
            source="env",
            components={
                "prefill": "flashattention4",
                "decode": "flashinfer",
                "kv_mode": "paged-materialized",
            },
            rationale="constructed in child",
            architecture={
                "arch": "cuda",
                "device_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                "sm": 120,
                "kernel_tier": "full",
            },
        )
        assert backend.kv_cache_dtype_requested == "auto"
        assert backend.kv_cache_dtype_resolved == "not-applicable"
        assert backend.generation_defaults.eos_token_id == 151645
        assert backend.generation_defaults.stop_token_ids == (151643,)
        assert backend.generation_defaults.sampling_defaults() == (
            _READY_GENERATION_DEFAULTS["sampling"]
        )
        assert backend.generation_defaults.mode == "auto"
        assert backend.generation_defaults.source == "generation_config.json"
    finally:
        await backend._reset_dead_locked()


async def test_startup_rejects_generation_mode_different_from_parent_request(
    monkeypatch,
):
    monkeypatch.setattr(
        "kairyu.engine.zmq_backend.run_engine_service",
        _fake_generation_mode_mismatch_engine_service,
    )
    backend = ZmqEngineBackend(num_pages=64, generation_config="auto")

    with pytest.raises(EngineServiceError, match="generation_config request mismatch"):
        await asyncio.to_thread(backend._spawn)

    assert backend._process is None


@pytest.mark.parametrize(
    "sampling",
    [
        {**_READY_GENERATION_DEFAULTS["sampling"], "temperature": True},
        {**_READY_GENERATION_DEFAULTS["sampling"], "top_p": float("nan")},
        {**_READY_GENERATION_DEFAULTS["sampling"], "top_k": True},
        {"temperature": 0.6},
    ],
)
async def test_generation_startup_metadata_is_strictly_validated(sampling):
    frame = {
        "startup_wire_version": STARTUP_WIRE_VERSION,
        "status": "ready",
        "generation_defaults": {
            **_READY_GENERATION_DEFAULTS,
            "sampling": sampling,
        },
    }

    with pytest.raises(EngineServiceError):
        _generation_defaults_from_startup_frame(frame)


async def test_startup_ready_frame_round_trips_generation_defaults():
    expected = GenerationDefaults(
        eos_token_id=151645,
        stop_token_ids=(151643,),
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.05,
        repetition_penalty=1.1,
        mode="auto",
        source="generation_config.json",
    )
    frame = _startup_ready_frame(
        types.SimpleNamespace(generation_defaults=expected)
    )

    assert _generation_defaults_from_startup_frame(frame) == expected


async def test_startup_ready_frame_round_trips_decode_graph_metadata():
    expected = {
        "decode_mode": "cuda_graph",
        "cuda_graph_decode": True,
        "cuda_graph_buckets": (1, 2, 4),
        "cuda_graph_captures": 3,
        "cuda_graph_replays": 7,
        "cuda_graph_eager_fallbacks": 2,
    }
    frame = _startup_ready_frame(
        types.SimpleNamespace(
            _runner=types.SimpleNamespace(
                decode_graph_metadata=lambda: expected
            )
        )
    )

    assert _decode_graph_metadata_from_wire(
        frame["decode_graph_metadata"]
    ) == expected


@pytest.mark.parametrize(
    "metadata",
    [
        {"decode_mode": "cuda_graph"},
        {
            "decode_mode": "cuda_graph",
            "cuda_graph_decode": True,
            "cuda_graph_buckets": [1, 4, 2],
            "cuda_graph_captures": 3,
            "cuda_graph_replays": 0,
            "cuda_graph_eager_fallbacks": 0,
        },
        {
            "decode_mode": "cuda_graph",
            "cuda_graph_decode": True,
            "cuda_graph_buckets": [1, 2],
            "cuda_graph_captures": 1,
            "cuda_graph_replays": 0,
            "cuda_graph_eager_fallbacks": 0,
        },
        {
            "decode_mode": "eager",
            "cuda_graph_decode": False,
            "cuda_graph_buckets": [],
            "cuda_graph_captures": 0,
            "cuda_graph_replays": 0,
            "cuda_graph_eager_fallbacks": -1,
        },
    ],
)
async def test_decode_graph_metadata_wire_validation_fails_closed(metadata):
    with pytest.raises(EngineServiceError, match="decode-graph metadata"):
        _decode_graph_metadata_from_wire(metadata)


async def test_decode_graph_metadata_snapshot_is_a_copy():
    backend = ZmqEngineBackend(num_pages=64)
    backend._live_generation = 23
    backend._decode_graph_metadata = {
        "decode_mode": "cuda_graph",
        "cuda_graph_decode": True,
        "cuda_graph_buckets": (1, 2),
        "cuda_graph_captures": 2,
        "cuda_graph_replays": 3,
        "cuda_graph_eager_fallbacks": 1,
    }

    snapshot = backend.decode_graph_metadata_snapshot()
    assert snapshot == backend._decode_graph_metadata
    assert snapshot is not backend._decode_graph_metadata
    assert backend.decode_graph_metric_generation_snapshot() == 23


async def test_submit_resolves_omissions_but_preserves_explicit_neutrals(
    monkeypatch,
):
    import msgpack

    class Socket:
        def __init__(self):
            self.messages = []

        async def send(self, payload):
            self.messages.append(msgpack.unpackb(payload))

    backend = ZmqEngineBackend(num_pages=64)
    backend.generation_defaults = GenerationDefaults(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.05,
        repetition_penalty=1.1,
    )
    socket = Socket()
    backend._socket = socket
    backend._live_generation = 1

    async def already_started():
        return None

    monkeypatch.setattr(backend, "_ensure_started", already_started)
    omitted = SamplingParams(max_tokens=1).with_generation_config_omitted(
        GENERATION_CONFIG_SAMPLING_FIELDS
    )

    await backend._submit(
        GenerationRequest("omitted", "hello", omitted)
    )
    await backend._submit(
        GenerationRequest("explicit", "hello", SamplingParams(max_tokens=1))
    )

    omitted_wire, explicit_wire = (
        message["sampling"] for message in socket.messages
    )
    assert {
        field: omitted_wire[field]
        for field in GENERATION_CONFIG_SAMPLING_FIELDS
    } == {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.05,
        "repetition_penalty": 1.1,
    }
    assert {
        field: explicit_wire[field]
        for field in GENERATION_CONFIG_SAMPLING_FIELDS
    } == {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    }


async def test_new_parent_accepts_a_legacy_child_with_only_the_port_frame(
    monkeypatch,
):
    monkeypatch.setattr(
        "kairyu.engine.zmq_backend.run_engine_service",
        _fake_legacy_engine_service,
    )
    backend = ZmqEngineBackend(num_pages=64)
    backend.attention_backend_decision = AttentionBackendDecision(
        requested="auto",
        resolved="torch",
        source="hw_profile",
        components={"prefill": "torch", "decode": "torch"},
        rationale="stale",
    )
    backend.generation_defaults = GenerationDefaults(
        temperature=0.6,
        source="stale-child",
    )
    try:
        port = await asyncio.to_thread(backend._spawn)
        assert port == 43211
        assert backend.attention_backend_decision is None
        assert backend.generation_defaults == backend._configured_generation_defaults
    finally:
        await backend._reset_dead_locked()


async def test_real_model_rejects_a_child_without_generation_metadata(
    monkeypatch,
):
    monkeypatch.setattr(
        "kairyu.engine.zmq_backend.run_engine_service",
        _fake_legacy_engine_service,
    )
    backend = ZmqEngineBackend(
        num_pages=64,
        model_path="/models/real",
    )

    with pytest.raises(EngineServiceError, match="omitted generation defaults"):
        await asyncio.to_thread(backend._spawn)

    assert backend._process is None


async def test_new_child_tolerates_a_legacy_parent_closing_after_the_port():
    parent_pipe, child_pipe = multiprocessing.get_context("spawn").Pipe()
    parent_pipe.close()
    try:
        assert (
            _send_optional_startup_frame(
                child_pipe,
                {
                    "startup_wire_version": STARTUP_WIRE_VERSION,
                    "status": "ready",
                    "attention_backend_decision": _READY_DECISION,
                },
            )
            is False
        )
    finally:
        child_pipe.close()


async def test_startup_failure_is_actionable_and_clears_a_stale_decision(
    monkeypatch,
):
    monkeypatch.setattr(
        "kairyu.engine.zmq_backend.run_engine_service",
        _fake_failing_engine_service,
    )
    backend = ZmqEngineBackend(num_pages=64)
    backend.attention_backend_decision = AttentionBackendDecision(
        requested="auto",
        resolved="torch",
        source="hw_profile",
        components={"prefill": "torch", "decode": "torch"},
        rationale="stale",
    )

    with pytest.raises(
        EngineServiceError,
        match="startup failed with RuntimeError: unsupported SM",
    ):
        await asyncio.to_thread(backend._spawn)

    assert backend.attention_backend_decision is None
    assert backend._process is None


async def test_real_child_build_failure_crosses_the_startup_pipe():
    backend = ZmqEngineBackend(num_pages=64, pipeline_depth=0)

    with pytest.raises(
        EngineServiceError,
        match="startup failed with ValueError: pipeline_depth must be >= 1",
    ):
        await backend.generate(_request("bad-startup", "prompt", max_tokens=1))

    assert backend._process is None
    assert backend.attention_backend_decision is None
    assert backend._active_request_ids == set()


async def test_cancelled_model_load_reclaims_its_child_before_a_new_generation(
    monkeypatch,
):
    monkeypatch.setattr(
        "kairyu.engine.zmq_backend.run_engine_service",
        _fake_slow_starting_engine_service,
    )
    backend = ZmqEngineBackend(num_pages=64)
    port_sent = multiprocessing.get_context("spawn").Event()
    backend._config["_test_port_sent"] = port_sent
    starting = asyncio.create_task(backend._ensure_started())
    assert await asyncio.to_thread(port_sent.wait, 3)
    # The first frame is now in the Pipe; let the parent consume it and enter
    # the unbounded model-load wait for the optional second frame.
    await asyncio.sleep(0.1)
    abandoned_process = backend._process
    assert abandoned_process is not None and abandoned_process.is_alive()
    backend.attention_backend_decision = _decision_from_startup_frame(
        {
            "startup_wire_version": STARTUP_WIRE_VERSION,
            "status": "ready",
            "attention_backend_decision": _READY_DECISION,
        }
    )

    starting.cancel()
    asyncio.get_running_loop().call_later(0.01, starting.cancel)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(starting, timeout=3)

    assert backend._process is None
    assert backend.attention_backend_decision is None
    assert not abandoned_process.is_alive()

    # A later generation owns the fields; the drained worker from above cannot
    # wake up after model load and overwrite this decision.
    monkeypatch.setattr(
        "kairyu.engine.zmq_backend.run_engine_service",
        _fake_ready_engine_service,
    )
    try:
        await backend._ensure_started()
        assert backend._process is not abandoned_process
        assert backend.attention_backend_decision is not None
        assert backend.attention_backend_decision.resolved == "flashattention4"
    finally:
        await backend._reset_dead_locked()


async def test_shutdown_before_spawn_registers_a_process_closes_admission(
    monkeypatch,
):
    backend = ZmqEngineBackend(num_pages=64)
    entered = threading.Event()
    release = threading.Event()
    original_spawn = backend._spawn

    def delayed_spawn(abandoned=None):
        entered.set()
        if not release.wait(3):
            raise RuntimeError("test did not release delayed spawn")
        return original_spawn(abandoned)

    monkeypatch.setattr(backend, "_spawn", delayed_spawn)
    starting = asyncio.create_task(backend._ensure_started())
    shutdown = None
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        shutdown = asyncio.create_task(backend.shutdown())
        for _ in range(300):
            abandoned = backend._startup_abandoned
            if backend._closed and abandoned is not None and abandoned.is_set():
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("shutdown did not close admission and cancel startup")

        release.set()
        with pytest.raises(EngineServiceError, match="startup was cancelled"):
            await asyncio.wait_for(starting, timeout=3)
        await asyncio.wait_for(shutdown, timeout=3)

        assert backend._process is None
        assert backend._startup_task is None
        assert backend._startup_abandoned is None
        with pytest.raises(EngineServiceError, match="shut down"):
            await backend._ensure_started()
    finally:
        release.set()
        if not starting.done():
            starting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await starting
        if shutdown is not None and not shutdown.done():
            await shutdown


async def test_shutdown_cancels_and_reaps_an_inflight_model_load(
    monkeypatch,
):
    monkeypatch.setattr(
        "kairyu.engine.zmq_backend.run_engine_service",
        _fake_slow_starting_engine_service,
    )
    backend = ZmqEngineBackend(num_pages=64)
    port_sent = multiprocessing.get_context("spawn").Event()
    backend._config["_test_port_sent"] = port_sent
    starting = asyncio.create_task(backend._ensure_started())

    assert await asyncio.to_thread(port_sent.wait, 3)
    await asyncio.sleep(0.1)
    process = backend._process
    assert process is not None and process.is_alive()

    shutdown = asyncio.create_task(backend.shutdown())
    with pytest.raises(EngineServiceError, match="startup was cancelled"):
        await asyncio.wait_for(starting, timeout=3)
    await asyncio.wait_for(shutdown, timeout=3)

    assert not process.is_alive()
    assert backend._process is None
    assert backend._startup_task is None
    assert backend._startup_abandoned is None


@pytest.mark.parametrize(
    "frame",
    (
        {
            "startup_wire_version": "1",
            "status": "ready",
            "attention_backend_decision": _READY_DECISION,
        },
        {
            "startup_wire_version": STARTUP_WIRE_VERSION,
            "status": "ready",
            "attention_backend_decision": {"resolved": "torch"},
        },
        {
            "startup_wire_version": STARTUP_WIRE_VERSION,
            "status": "ready",
            "attention_backend_decision": {
                **_READY_DECISION,
                "components": {"prefill": 4},
            },
        },
        {
            "startup_wire_version": STARTUP_WIRE_VERSION,
            "status": "ready",
            "attention_backend_decision": {
                **_READY_DECISION,
                "architecture": {4: "cuda"},
            },
        },
        {
            "startup_wire_version": STARTUP_WIRE_VERSION,
            "status": "ready",
            "attention_backend_decision": {
                **_READY_DECISION,
                "architecture": {"sm": {120}},
            },
        },
    ),
)
async def test_startup_metadata_is_type_checked_before_parent_retains_it(frame):
    with pytest.raises(EngineServiceError):
        _decision_from_startup_frame(frame)


async def test_reset_and_shutdown_clear_the_actual_child_decision():
    backend = ZmqEngineBackend(num_pages=64)
    decision = _decision_from_startup_frame(
        {
            "startup_wire_version": STARTUP_WIRE_VERSION,
            "status": "ready",
            "attention_backend_decision": _READY_DECISION,
        }
    )
    assert decision is not None
    backend.attention_backend_decision = decision
    await backend._reset_dead_locked()
    assert backend.attention_backend_decision is None

    backend.attention_backend_decision = decision
    await backend.shutdown()
    assert backend.attention_backend_decision is None


async def test_idle_generation_reset_does_not_retain_a_failure_marker():
    backend = ZmqEngineBackend(num_pages=64)
    backend._live_generation = 7

    await backend._reset_dead_locked()

    assert backend._live_generation is None
    assert backend._failed_generations == set()


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
        queue = await backend._submit(_request("r2", "in flight", max_tokens=10_000))
        backend.attention_backend_decision = _decision_from_startup_frame(
            {
                "startup_wire_version": STARTUP_WIRE_VERSION,
                "status": "ready",
                "attention_backend_decision": _READY_DECISION,
            }
        )
        backend._process.kill()
        # A token frame can already be queued before kill() takes effect. Wait
        # for the terminal death event rather than treating the first frame as
        # proof that the receiver has observed and cleared the dead child.
        async with asyncio.timeout(10):
            while "error" not in await queue.get():
                pass
        assert backend.attention_backend_decision is None
        backend._queues.pop("r2", None)
        backend._release_wire_route("r2")
        # the next request respawns a fresh child and completes normally
        result = await asyncio.wait_for(
            backend.generate(_request("r3", "recovered", max_tokens=2)), timeout=15
        )
        assert result.completions[0].token_ids
    finally:
        await backend.shutdown()


async def test_respawn_before_receiver_death_tick_fails_old_generation_once(
    monkeypatch,
):
    # Make the receiver's own death observation deliberately slow. A new
    # request must still reset the dead child without cancelling away the only
    # terminal event for an older in-flight request.
    monkeypatch.setattr("kairyu.engine.zmq_backend._RECV_TICK_S", 60.0)
    backend = ZmqEngineBackend(num_pages=64)
    queue = None
    try:
        queue = await asyncio.wait_for(
            backend._submit(_request("death-reset-race", "in flight", max_tokens=10_000)),
            timeout=15,
        )
        old_generation = backend._queue_generations["death-reset-race"]
        old_process = backend._process
        assert old_process is not None
        old_process.kill()
        await asyncio.to_thread(old_process.join, 3)
        assert not old_process.is_alive()

        # This used to cancel the old receiver before its timeout without
        # notifying the old queue, leaving that request blocked forever.
        await asyncio.wait_for(backend._ensure_started(), timeout=10)
        assert backend._live_generation != old_generation

        queued = []
        while True:
            try:
                queued.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        errors = [event for event in queued if "error" in event]
        assert len(errors) == 1
        assert "reset before completing the request" in errors[0]["error"]

        result = await asyncio.wait_for(
            backend.generate(_request("after-death-reset", "recovered", max_tokens=2)),
            timeout=10,
        )
        assert result.finished is True
    finally:
        if queue is not None:
            backend._queues.pop("death-reset-race", None)
            backend._release_wire_route("death-reset-race")
        await backend.shutdown()


async def test_shutdown_is_clean_and_idempotent():
    backend = ZmqEngineBackend(num_pages=64)
    await backend.generate(_request("x1", "before shutdown", max_tokens=2))
    process = backend._process
    await backend.shutdown()
    assert process is not None and not process.is_alive()
    assert process.exitcode == 0  # clean exit: coverage flushed
    await backend.shutdown()  # idempotent


async def test_shutdown_control_send_is_bounded_and_still_closes_transport(
    monkeypatch,
):
    class StoppedProcess:
        pid = None

        def __init__(self) -> None:
            self.join_calls = []

        def join(self, timeout=None):
            self.join_calls.append(timeout)

        def is_alive(self):
            return False

    monkeypatch.setattr(
        "kairyu.engine.zmq_backend._CONTROL_SEND_TIMEOUT_S",
        0.01,
    )
    backend = ZmqEngineBackend(num_pages=64)
    socket = _NeverSendingSocket()
    process = StoppedProcess()
    backend._socket = socket
    backend._process = process

    shutdown = asyncio.create_task(backend.shutdown())
    await asyncio.wait_for(asyncio.shield(shutdown), timeout=0.5)

    assert shutdown.done() and not shutdown.cancelled()
    assert shutdown.exception() is None
    assert socket.send_started is True
    assert socket.send_cancelled is True
    assert socket.closed is True
    assert process.join_calls
    assert backend._process is None


async def test_cancelled_shutdown_drains_one_shared_reap_for_all_callers(
    monkeypatch,
):
    class LiveProcess:
        pid = None

        def is_alive(self):
            return True

    reap_entered = threading.Event()
    release_reap = threading.Event()
    reap_calls = []

    def controlled_reap(process, timeout_s):
        reap_calls.append((process, timeout_s))
        reap_entered.set()
        assert release_reap.wait(timeout=2)
        return True

    monkeypatch.setattr(
        "kairyu.engine.zmq_backend._wait_process_tree_exit",
        controlled_reap,
    )
    backend = ZmqEngineBackend(num_pages=64)
    process = LiveProcess()
    backend._process = process
    first = asyncio.create_task(backend.shutdown())
    second = None
    try:
        assert await asyncio.to_thread(reap_entered.wait, 1)
        shared_shutdown = backend._shutdown_task
        assert shared_shutdown is not None
        second = asyncio.create_task(backend.shutdown())
        await asyncio.sleep(0)

        first.cancel()
        await asyncio.sleep(0)
        assert first.done() is False
        assert second.done() is False
        release_reap.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(first, timeout=1)
        await asyncio.wait_for(second, timeout=1)
    finally:
        release_reap.set()
        for caller in (first, second):
            if caller is not None and not caller.done():
                with contextlib.suppress(asyncio.CancelledError):
                    await caller

    assert reap_calls == [(process, 5.0)]
    assert backend._process is None
    assert backend._shutdown_task is shared_shutdown
    assert shared_shutdown.done() and shared_shutdown.exception() is None

    await backend.shutdown()
    assert backend._shutdown_task is shared_shutdown
    assert reap_calls == [(process, 5.0)]


async def test_failed_shutdown_cleanup_can_be_retried(monkeypatch):
    class LiveProcess:
        pid = None

        def is_alive(self):
            return True

    reap_calls = []
    signals = []

    def fail_then_succeed(process, timeout_s):
        reap_calls.append((process, timeout_s))
        return len(reap_calls) >= 4

    monkeypatch.setattr(
        "kairyu.engine.zmq_backend._wait_process_tree_exit",
        fail_then_succeed,
    )
    monkeypatch.setattr(
        "kairyu.engine.zmq_backend._signal_process_tree",
        lambda _process, sig: signals.append(sig),
    )
    backend = ZmqEngineBackend(num_pages=64)
    process = LiveProcess()
    backend._process = process

    with pytest.raises(EngineServiceError, match="not completely reaped"):
        await backend.shutdown()

    assert backend._shutdown_task is None
    assert backend._process is process
    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert [timeout for _, timeout in reap_calls] == [5.0, 2.0, 2.0]

    await backend.shutdown()
    successful_shutdown = backend._shutdown_task
    assert successful_shutdown is not None and successful_shutdown.done()
    assert successful_shutdown.exception() is None
    assert backend._process is None
    assert [timeout for _, timeout in reap_calls] == [5.0, 2.0, 2.0, 5.0]

    await backend.shutdown()
    assert backend._shutdown_task is successful_shutdown
    assert len(reap_calls) == 4


async def test_abort_control_send_is_bounded_and_returns_normally(monkeypatch):
    monkeypatch.setattr(
        "kairyu.engine.zmq_backend._CONTROL_SEND_TIMEOUT_S",
        0.01,
    )
    backend = ZmqEngineBackend(num_pages=64)
    socket = _NeverSendingSocket()
    backend._socket = socket
    backend._wire_request_ids["bounded-abort"] = "wire-bounded-abort"
    backend._stream_ids["bounded-abort"] = "stream-bounded-abort"

    abort = asyncio.create_task(backend._abort("bounded-abort"))
    await asyncio.wait_for(asyncio.shield(abort), timeout=0.5)

    assert abort.done() and not abort.cancelled()
    assert abort.exception() is None
    assert socket.send_started is True
    assert socket.send_cancelled is True
    backend._socket = None


async def test_submit_send_is_bounded_and_releases_its_route(monkeypatch):
    monkeypatch.setattr(
        "kairyu.engine.zmq_backend._CONTROL_SEND_TIMEOUT_S",
        0.01,
    )
    backend = ZmqEngineBackend(num_pages=64)
    socket = _NeverSendingSocket()
    backend._socket = socket
    backend._live_generation = 9

    async def already_started():
        return None

    monkeypatch.setattr(backend, "_ensure_started", already_started)
    with pytest.raises(EngineServiceError, match="request send timed out"):
        await asyncio.wait_for(
            backend._submit(_request("bounded-add", "prompt", max_tokens=2)),
            timeout=0.5,
        )

    assert socket.send_started is True
    assert socket.send_cancelled is True
    assert "bounded-add" not in backend._queues
    assert "bounded-add" not in backend._wire_request_ids
    assert "bounded-add" not in backend._queue_generations
    assert backend.readiness().fatal is True
    backend._socket = None


async def test_submit_does_not_send_old_frame_to_replacement_generation(
    monkeypatch,
):
    class CapturingSocket:
        def __init__(self):
            self.payloads = []

        async def send(self, payload):
            self.payloads.append(payload)

    backend = ZmqEngineBackend(num_pages=64)
    backend.generation_defaults = GenerationDefaults()
    old_socket = CapturingSocket()
    new_socket = CapturingSocket()
    backend._socket = old_socket
    backend._live_generation = 1
    pack_started = threading.Event()
    release_pack = threading.Event()
    original_pack = backend._pack_add_message

    async def already_started():
        return None

    def blocked_pack(*args):
        pack_started.set()
        assert release_pack.wait(timeout=5)
        return original_pack(*args)

    monkeypatch.setattr(backend, "_ensure_started", already_started)
    monkeypatch.setattr(backend, "_pack_add_message", blocked_pack)
    submit = asyncio.create_task(
        backend._submit(_request("generation-swap", "prompt", max_tokens=2))
    )
    assert await asyncio.to_thread(pack_started.wait, 5)

    backend._socket = new_socket
    backend._live_generation = 2
    release_pack.set()

    with pytest.raises(EngineServiceError, match="changed while packing"):
        await submit
    assert old_socket.payloads == []
    assert new_socket.payloads == []
    assert "generation-swap" not in backend._queues
    assert "generation-swap" not in backend._wire_request_ids
    assert backend._live_generation == 2
    backend._socket = None
    backend._live_generation = None
    await backend.shutdown()


async def test_cancelled_submit_send_retires_unknown_delivery_generation(monkeypatch):
    backend = ZmqEngineBackend(num_pages=64)
    socket = _NeverSendingSocket()
    backend._socket = socket
    backend._live_generation = 10

    async def already_started():
        return None

    monkeypatch.setattr(backend, "_ensure_started", already_started)
    submit = asyncio.create_task(
        backend._submit(_request("cancelled-add", "prompt", max_tokens=2))
    )
    # The prompt worker thread needs wall-clock time before the send starts;
    # bare loop yields are not enough on a loaded runner.
    for _ in range(500):
        if socket.send_started:
            break
        await asyncio.sleep(0.01)
    assert socket.send_started is True

    submit.cancel()
    with pytest.raises(asyncio.CancelledError):
        await submit

    assert socket.send_cancelled is True
    assert "cancelled-add" not in backend._queues
    assert "cancelled-add" not in backend._wire_request_ids
    assert "cancelled-add" not in backend._queue_generations
    status = backend.readiness()
    assert status.ready is False and status.fatal is True
    backend._socket = None


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
    duplicate_stream = zmq_backend.stream(_request("same", "second stream", max_tokens=2))
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
        proc_status = Path(f"/proc/{process.pid}/status")
        for _ in range(500):
            state_line = next(
                (
                    line
                    for line in proc_status.read_text(encoding="utf-8").splitlines()
                    if line.startswith("State:")
                ),
                "",
            )
            if state_line.split()[1:2] in (["T"], ["t"]):
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("engine service did not enter a stopped process state")
        original.cancel()
        with pytest.raises(asyncio.CancelledError):
            await original

        reused = asyncio.create_task(
            backend.generate(_request("queued-reuse", "reused", max_tokens=2))
        )
        for _ in range(500):
            if "queued-reuse" in backend._active_request_ids and "queued-reuse" in backend._queues:
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
    backend._live_generation = 1

    with pytest.raises(RuntimeError, match="send failed"):
        await backend.generate(_request("send-failure", "prompt", max_tokens=2))

    assert "send-failure" not in backend._active_request_ids
    assert "send-failure" not in backend._queues
    assert "send-failure" not in backend._queue_generations


@pytest.mark.parametrize("original_api", ["generate", "stream"])
async def test_duplicate_request_id_stays_reserved_until_abort_finishes(monkeypatch, original_api):
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
            await backend.generate(_request("abort-race", "duplicate generate", max_tokens=2))
        duplicate_stream = backend.stream(_request("abort-race", "duplicate stream", max_tokens=2))
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
