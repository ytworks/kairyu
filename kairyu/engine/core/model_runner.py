"""PagedModelRunner: the real ModelRunner over DenseDecoder + PagedKVPool (m12 D4).

State-access contract (canonical, m12 review): reads exactly
``request.{prompt_token_ids, request_id, sampling, eos_token_id}``,
``allocation.pages``, ``allocation.num_cached_tokens``, ``decode_pages``,
``computed_prompt``, ``prefill_done``, ``outputs`` (values). The decode input
token comes from the PASSED state (`outputs[p-1]`) at execute time — that is
what keeps SpeculativeRunner's overlay mechanism working unchanged. KV is
written before it is read at every decode position; positions below
``num_cached_tokens`` are never rewritten (shared radix slots).
Requests use tensor batching when the model/attention backend supports it and
retain the sequential compatibility path otherwise.

m2 §2.2 status (the "future token" technique), so the scope is stated where the
code is rather than only in the design doc:

- DONE — the decode input slots are persistent device tensors written in place
  (`_decode_input_slots`), shared by the batched and the single-request path, so
  no decode step allocates a device tensor and neither path is a fallback.
- DONE (#136) — the runner keeps its uncommitted tokens, so overlap can resolve
  a decode input one step before the scheduler commits it.
- DONE (#206) — grammar-free CUDA sampling (greedy, filtered stochastic,
  penalties, and logprobs) produces a device scalar. The next decode input is
  patched D2D and the measured runner feedback interval contains no
  ``.item()``, ``.cpu()``, or event synchronization. Public token/logprob
  materialization is batched onto a copy stream and resolved one step late at
  the host EOS/streaming boundary.
- COMPATIBILITY — xgrammar's stateful matcher remains on the CPU, so structured
  requests use the reviewed mask/accept path rather than a stale device mask.
- DONE (#207) — supported eager and captured batched-decode attention share
  the tensor metadata path; KV-write masking and ragged page tables never read
  a per-row device scalar into Python.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass

import torch

from kairyu.engine.core.attention import graph_capture_gap
from kairyu.engine.core.kv_pool import PagedKVPool
from kairyu.engine.core.prefill import (
    PrefillSequence,
    build_prefill_batch,
)
from kairyu.engine.core.sampler import DeviceSample, Sampler
from kairyu.engine.core.sampling_types import SampledToken
from kairyu.engine.core.scheduler import ScheduledChunk
from kairyu.models.llama import DenseDecoder


@dataclass
class _PendingDeviceToken:
    sample: DeviceSample
    on_resolve: Callable[[SampledToken], None]
    host_token: torch.Tensor | None = None
    host_logprob: torch.Tensor | None = None
    host_top_indices: torch.Tensor | None = None
    host_top_logprobs: torch.Tensor | None = None


class _DeferredStepOutput(Mapping[str, tuple[SampledToken, ...]]):
    """Step output whose small D2H copies resolve at the late commit boundary.

    ``PagedModelRunner.execute`` only enqueues these copies.  Under
    ``OverlapEngineCore`` the next model step is submitted before
    ``token_ids()`` indexes this mapping, so the future-token device slot is
    already in use while EOS/stop/streaming state resolves one step late.
    """

    def __init__(
        self,
        records: Mapping[
            str, tuple[SampledToken | _PendingDeviceToken, ...]
        ],
        copy_stream: torch.cuda.Stream | None = None,
    ) -> None:
        self._records = dict(records)
        self._resolved: dict[str, tuple[SampledToken, ...]] | None = None
        self._event: torch.cuda.Event | None = None
        pending = [
            record
            for values in self._records.values()
            for record in values
            if isinstance(record, _PendingDeviceToken)
        ]
        has_cuda = bool(
            pending and pending[0].sample.token_id.device.type == "cuda"
        )

        # One token vector per step, not B scalar D2H operations.  The vector is
        # immutable and retained by this output, so a dedicated copy stream can
        # run independently of the next model step without an overwrite race.
        device_tokens = (
            torch.stack([record.sample.token_id for record in pending])
            if pending
            else None
        )
        host_tokens = (
            torch.empty(
                len(pending),
                dtype=torch.long,
                device="cpu",
                pin_memory=has_cuda,
            )
            if pending
            else None
        )
        for index, record in enumerate(pending):
            assert host_tokens is not None
            record.host_token = host_tokens[index]

        device_logprobs = [
            record.sample.logprob
            for record in pending
            if record.sample.logprob is not None
        ]
        device_logprob_tensor = (
            torch.stack(device_logprobs) if device_logprobs else None
        )
        # These stacked staging tensors are the actual async-copy sources. The
        # scalar DeviceSamples retained in ``_records`` do not retain a stack's
        # separate storage. Keep it alive until the copy event is synchronized;
        # otherwise an immediately allocated TP token packet can reuse the
        # storage and overwrite a public token with its ``-1`` sentinel.
        self._copy_sources = tuple(
            source
            for source in (device_tokens, device_logprob_tensor)
            if source is not None
        )
        host_logprobs = (
            torch.empty(
                len(device_logprobs),
                dtype=torch.float32,
                device="cpu",
                pin_memory=has_cuda,
            )
            if device_logprobs
            else None
        )
        logprob_index = 0
        for record in pending:
            if record.sample.logprob is not None:
                assert host_logprobs is not None
                record.host_logprob = host_logprobs[logprob_index]
                logprob_index += 1

        def enqueue_copies() -> None:
            if device_tokens is not None and host_tokens is not None:
                host_tokens.copy_(device_tokens, non_blocking=has_cuda)
            if device_logprob_tensor is not None and host_logprobs is not None:
                host_logprobs.copy_(
                    device_logprob_tensor, non_blocking=has_cuda
                )
            for record in pending:
                sample = record.sample
                if sample.top_indices is not None:
                    record.host_top_indices = torch.empty_like(
                        sample.top_indices, device="cpu", pin_memory=has_cuda
                    )
                    record.host_top_indices.copy_(
                        sample.top_indices, non_blocking=has_cuda
                    )
                if sample.top_logprobs is not None:
                    record.host_top_logprobs = torch.empty_like(
                        sample.top_logprobs, device="cpu", pin_memory=has_cuda
                    )
                    record.host_top_logprobs.copy_(
                        sample.top_logprobs, non_blocking=has_cuda
                    )

        for values in self._records.values():
            for record in values:
                if isinstance(record, _PendingDeviceToken):
                    # Keep all source tensors alive through the async transfer.
                    _ = record.sample
        if has_cuda:
            if copy_stream is None:
                raise ValueError("CUDA deferred output requires a copy stream")
            producer = torch.cuda.current_stream(device_tokens.device)
            copy_stream.wait_stream(producer)
            with torch.cuda.stream(copy_stream):
                enqueue_copies()
                self._event = torch.cuda.Event()
                self._event.record(copy_stream)
        else:
            enqueue_copies()
            self._resolve()

    def ready(self) -> bool:
        return self._resolved is not None or self._event is None or self._event.query()

    def _resolve(self) -> dict[str, tuple[SampledToken, ...]]:
        if self._resolved is not None:
            return self._resolved
        if self._event is not None:
            self._event.synchronize()
        resolved: dict[str, tuple[SampledToken, ...]] = {}
        for request_id, values in self._records.items():
            host_values: list[SampledToken] = []
            for record in values:
                if isinstance(record, SampledToken):
                    host_values.append(record)
                    continue
                assert record.host_token is not None
                token_id = int(record.host_token.item())
                logprob = (
                    None
                    if record.host_logprob is None
                    else float(record.host_logprob.item())
                )
                top = None
                if (
                    record.host_top_indices is not None
                    and record.host_top_logprobs is not None
                ):
                    top = tuple(
                        (int(index), float(value))
                        for index, value in zip(
                            record.host_top_indices.tolist(),
                            record.host_top_logprobs.tolist(),
                            strict=True,
                        )
                    )
                token = SampledToken(token_id, logprob, top)
                record.on_resolve(token)
                host_values.append(token)
            resolved[request_id] = tuple(host_values)
        self._resolved = resolved
        self._records.clear()
        self._copy_sources = ()
        self._event = None
        return resolved

    def __getitem__(self, key: str) -> tuple[SampledToken, ...]:
        return self._resolve()[key]

    def __iter__(self) -> Iterator[str]:
        source = self._resolved if self._resolved is not None else self._records
        return iter(source)

    def __len__(self) -> int:
        source = self._resolved if self._resolved is not None else self._records
        return len(source)

    def items(self):
        return self._resolve().items()

    def values(self):
        return self._resolve().values()

    def raw_records(
        self,
    ) -> Mapping[str, tuple[SampledToken | _PendingDeviceToken, ...]]:
        """Return token records without forcing the deferred D2H boundary.

        Tensor-parallel rank 0 needs the already-sampled device ids for the
        canonical-token broadcast.  Resolving this mapping here would put the
        public host materialization back on the critical decode dependency
        chain that #206 removed.
        """
        if self._resolved is not None:
            return self._resolved
        return self._records


def _tensor_decode_gap(model: DenseDecoder) -> str | None:
    """Why this model cannot run the tensor decode contract, or None (m17 [P2]).

    ``_graph_decode`` calls ``forward_decode_tensors``, which unconditionally
    calls ``backend.attend_decode``. ``MlaAttention`` has no tensor decode form
    at all, so an MLA model used to construct fine and then die with an
    ``AttributeError`` on the first BATCHED decode, arbitrarily far into a run.

    Presence of ``attend_decode`` is NOT the question, though (review [P1]).
    A backend can own that method and still be uncapturable, because what
    breaks a capture is host synchronization inside it — FlashInfer's
    ``plan()`` copies ``indptr`` to the CPU and cannot run under capture at
    all. Checking the method existed would have let such a backend construct
    and then fail on the FIRST capture with a D2H ``RuntimeError``, which is
    exactly the late failure this gate exists to prevent. So the question is
    the declared ``GraphDecodeBackend`` capability, enforced in one place by
    ``graph_capture_gap``.

    Checked once, at construction, where the operator can still act on it.
    """
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return f"{type(model).__name__} exposes no model.layers to check"
    if not callable(getattr(model, "plan_decode_tensors", None)):
        return (
            f"{type(model).__name__} has no plan_decode_tensors, so the step "
            "boundary never reaches its attention backends"
        )
    for index, layer in enumerate(layers):
        attention = getattr(layer, "self_attn", None)
        if not hasattr(attention, "forward_decode_tensors"):
            return (
                f"layer {index}'s attention ({type(attention).__name__}) has no "
                "forward_decode_tensors"
            )
        gap = graph_capture_gap(getattr(attention, "backend", None))
        if gap is not None:
            return f"layer {index}'s attention backend {gap}"
    return None


def _batched_prefill_gap(model: DenseDecoder) -> str | None:
    """Why this model/backend must retain sequential prefill, or ``None``.

    ``attend_batched`` alone is not enough: the Torch compatibility backend
    implements it as a Python row loop.  The fast path requires an explicit
    native flat-query contract so a claimed batch really removes the
    request-proportional model and attention launch chains.
    """
    if not callable(getattr(model, "forward_prefill_batch", None)):
        return f"{type(model).__name__} has no forward_prefill_batch"
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return f"{type(model).__name__} exposes no model.layers to check"
    for index, layer in enumerate(layers):
        attention = getattr(layer, "self_attn", None)
        if not callable(getattr(attention, "forward_prefill_batch", None)):
            return (
                f"layer {index}'s attention ({type(attention).__name__}) has no "
                "forward_prefill_batch"
            )
        backend = getattr(attention, "backend", None)
        if not getattr(backend, "supports_batched_prefill", False):
            return (
                f"layer {index}'s attention backend "
                f"{type(backend).__name__} does not declare "
                "supports_batched_prefill"
            )
        if not callable(getattr(backend, "attend_prefill", None)):
            return (
                f"layer {index}'s attention backend "
                f"{type(backend).__name__} declares supports_batched_prefill "
                "but has no attend_prefill"
            )
    return None


class PagedModelRunner:
    def __init__(
        self,
        model: DenseDecoder,
        pool: PagedKVPool,
        sampler: Sampler | None = None,
        cache: object | None = None,
        graph_backend: object | None = None,
        graph_max_batch: int = 0,
        graph_max_pages: int = 0,
        graph_scratch_page: int | None = None,
        sampling_owner: bool = True,
        enable_batched_prefill: bool = True,
    ) -> None:
        if cache is not None:  # fail-fast sizing agreement (m12 D3)
            if pool.num_pages != cache.num_pages or pool.page_size != cache.page_size:
                raise ValueError(
                    f"pool ({pool.num_pages} pages x {pool.page_size}) disagrees with "
                    f"cache ({cache.num_pages} x {cache.page_size})"
                )
        if pool.num_layers != model.config.num_hidden_layers:
            raise ValueError("pool layer count disagrees with the model config")
        self._model = model
        self._pool = pool
        self._sampler = sampler
        self._sampling_owner = sampling_owner
        # Input tensors (token ids, positions) must be built on the model's device
        # so the GPU forward never mixes CPU inputs with on-device weights/KV.
        self._device = next(model.parameters()).device
        # In-flight tokens: under OverlapEngineCore the snapshot for step N+1 is
        # taken BEFORE step N's token is committed, so `state.outputs` is short
        # and reading `outputs[position - 1]` raised IndexError. The runner
        # already produced those tokens; it just never kept them.
        #
        # Host copies remain the compatibility history for CPU/structured
        # sampling and for late public output materialization.
        self._future_tokens: dict[str, dict[int, int]] = {}
        # CUDA sampling owns the same history as device scalars.  Decode consumes
        # position N directly when building input N+1; host materialization is
        # deliberately not on that dependency chain.
        self._future_device_tokens: dict[str, dict[int, torch.Tensor]] = {}
        # Rank 0 may outlive a public StepOutput while its pinned D2H copy is
        # still in flight under schedule-ahead. Retain such outputs until a
        # non-blocking event query says the transfer completed. Passive TP
        # followers never construct a StepOutput or enqueue this copy.
        self._deferred_outputs: deque[_DeferredStepOutput] = deque()
        self._output_copy_stream = (
            torch.cuda.Stream(device=self._device)
            if self._device.type == "cuda"
            else None
        )
        # The decode input slots themselves: allocated once on the device and
        # written IN PLACE every step (`_decode_input_slots`).
        self._decode_slots: torch.Tensor | None = None
        self._decode_positions: torch.Tensor | None = None
        self._slot_staging: torch.Tensor | None = None
        self._slot_copy_done: torch.cuda.Event | None = None
        self._slot_staging_pool: list[
            tuple[torch.Tensor, torch.cuda.Event]
        ] = []
        # The same tensor-only decode contract serves both graph capture and
        # eager batching. Unsupported model/attention combinations retain the
        # list-based compatibility path.
        self._tensor_decode_supported = _tensor_decode_gap(model) is None
        # Cross-request prefill is deliberately stricter than semantic
        # ``attend_batched`` support. Only a backend that owns one native
        # ragged plan/run opts in; Torch, MLA, and custom models keep the
        # established sequential behavior.
        self._prefill_batch_gap = _batched_prefill_gap(model)
        self._batched_prefill_enabled = bool(enable_batched_prefill)
        self._prefill_rows_executed = 0
        self._prefill_model_calls = 0
        self._prefill_batched_groups = 0
        self._prefill_sequential_rows = 0

        # m17 D1 / runbook §6.3: decode capture. OFF unless a backend is passed —
        # the seam has existed since m17 with no caller, and enabling it by
        # default would change what every deployment executes.
        self._graph = None
        self._graph_scratch_page: int | None = None
        if graph_backend is None and graph_scratch_page is not None:
            raise ValueError("graph_scratch_page requires a graph_backend")
        if graph_backend is not None:
            from kairyu.engine.core.step_executor import GraphStepExecutor

            if graph_max_batch < 1 or graph_max_pages < 1:
                raise ValueError(
                    "graph_max_batch and graph_max_pages must be >= 1 when a "
                    f"graph backend is given (got {graph_max_batch}, {graph_max_pages})"
                )
            gap = _tensor_decode_gap(model)
            if gap is not None:  # [P2]: fail here, not on the first batched decode
                raise ValueError(
                    f"graph decode needs the tensor decode contract but {gap}. "
                    "A backend must declare supports_graph_capture and provide "
                    "a plan_decode step hook (see GraphDecodeBackend) — build "
                    "this runner without graph_backend."
                )
            # [P1]: the padding rows of every replay write KV somewhere. That
            # page has to come OUT of the scheduler's allocator, so take it from
            # the cache the scheduler allocates from — there is no way to pick a
            # safe id without it.
            if graph_scratch_page is None:
                if cache is None:
                    raise ValueError(
                        "a graph backend needs either the scheduler's "
                        "RadixKVCache or an already-reserved scratch page via "
                        "graph_scratch_page: "
                        "captured padding rows write KV to a page the allocator "
                        "must never return (m17 A5)"
                    )
                graph_scratch_page = cache.reserve_scratch_page()
            elif not 0 <= graph_scratch_page < pool.num_pages:
                raise ValueError(
                    f"graph_scratch_page={graph_scratch_page} is outside the "
                    f"KV pool's [0, {pool.num_pages}) range"
                )
            elif cache is not None:
                reserved = cache.reserve_scratch_page()
                if reserved != graph_scratch_page:
                    raise ValueError(
                        f"graph_scratch_page={graph_scratch_page} disagrees with "
                        f"the scheduler cache's reserved page {reserved}"
                    )
            self._graph_scratch_page = graph_scratch_page
            self._graph = GraphStepExecutor(
                self._graph_decode,
                graph_backend,
                max_batch=graph_max_batch,
                max_pages=graph_max_pages,
                scratch_page=self._graph_scratch_page,
                device=self._device,
                plan_fn=self._plan_graph_decode,
            )

    def _plan_graph_decode(self, batch) -> None:
        """Step-boundary host phase, run OUTSIDE the captured region ([P1]).

        The executor calls this before it captures and again after every
        copy-in, so the attention backends plan against the very buffers the
        next ``replay()`` will read. Without it a planning backend such as
        FlashInfer has no live plan at capture time, and every later replay
        would attend over the pages that happened to be in the static buffers
        when the graph was recorded.
        """
        self._model.plan_decode_tensors(self._pool, batch.page_tables, batch.seq_lens)

    def _graph_decode(self, batch) -> torch.Tensor:
        """The captured region: embed -> layers -> norm -> logits, tensors only."""
        hidden = self._model.forward_decode_tensors(
            batch.token_ids, batch.positions, self._pool,
            batch.page_tables, batch.seq_lens, batch.write_from,
        )
        return self._model.logits(hidden)

    def invalidate_graphs(self) -> None:
        """Weight swap or pool resize: every capture is stale (m17 D2)."""
        if self._graph is not None:
            self._graph.invalidate()

    @property
    def sampler(self) -> Sampler | None:
        """The per-request sampling state a P-D handoff has to carry across."""
        return self._sampler

    @property
    def sampling_owner(self) -> bool:
        """Whether this runner is allowed to advance sampling state."""
        return getattr(self, "_sampling_owner", True)

    def set_batched_prefill_enabled(self, enabled: bool) -> None:
        """Switch the optimization without changing scheduling semantics.

        Primarily an operational rollback and matched-A/B evidence seam.  A
        capability gap still wins: enabling cannot make an unsupported backend
        enter the native path.
        """
        if type(enabled) is not bool:
            raise TypeError("batched prefill enabled flag must be bool")
        self._batched_prefill_enabled = enabled

    def prefill_execution_stats(self, *, reset: bool = False) -> dict[str, object]:
        """Return structural counters; optional reset is out-of-band only."""
        backend_rows: list[dict[str, object]] = []
        seen: set[int] = set()
        layers = getattr(getattr(self._model, "model", None), "layers", ())
        for layer in layers:
            backend = getattr(getattr(layer, "self_attn", None), "backend", None)
            if backend is None or id(backend) in seen:
                continue
            seen.add(id(backend))
            getter = getattr(backend, "prefill_execution_stats", None)
            if callable(getter):
                row = getter(reset=reset)
                if not isinstance(row, dict):
                    raise RuntimeError(
                        f"{type(backend).__name__}.prefill_execution_stats "
                        "must return a dict"
                    )
                backend_rows.append(row)
        result = {
            "enabled": getattr(self, "_batched_prefill_enabled", True),
            "capability_gap": getattr(self, "_prefill_batch_gap", None),
            "rows": getattr(self, "_prefill_rows_executed", 0),
            "model_calls": getattr(self, "_prefill_model_calls", 0),
            "batched_groups": getattr(self, "_prefill_batched_groups", 0),
            "sequential_rows": getattr(self, "_prefill_sequential_rows", 0),
            "backend": backend_rows,
        }
        if reset:
            self._prefill_rows_executed = 0
            self._prefill_model_calls = 0
            self._prefill_batched_groups = 0
            self._prefill_sequential_rows = 0
        return result

    def _require_sampling_owner(self) -> None:
        if not self.sampling_owner:
            raise RuntimeError(
                "this tensor-parallel rank is a passive sampling follower; "
                "only rank 0 may sample"
            )

    def release(self, request_id: str) -> None:
        """Drop per-request sampler state (seeds + grammar enforcer) on finish (E2)."""
        if self._sampler is not None:
            self._sampler.release(request_id)
        self._future_tokens.pop(request_id, None)
        self._future_device_tokens.pop(request_id, None)

    def _sample(self, state: object, logits: torch.Tensor, position: int) -> SampledToken:
        self._require_sampling_owner()
        if self._sampler is None:
            return SampledToken(int(torch.argmax(logits).item()))
        return self._sampler.sample(
            state.request.sampling_identity,
            state.request.sampling,
            position,
            logits,
            prompt=state.request.prompt_token_ids,
            # the SAME completion the decode input came from: presence,
            # frequency and repetition penalties are computed over this history,
            # so an overlap snapshot missing the in-flight token would penalise a
            # different set of tokens than the eager path and pick differently
            outputs=state.outputs,
            pending_outputs=self._pending_host_outputs(state, position),
            history_epoch=getattr(state, "output_epoch", 0),
            eos_token_id=state.request.eos_token_id,
        )

    def _pending_host_outputs(
        self, state: object, position: int
    ) -> tuple[int, ...]:
        """Uncommitted CPU completion history, bounded by overlap depth."""
        committed = len(state.outputs)
        if position <= committed:
            return ()
        pending = self._future_tokens.get(state.request.request_id, {})
        values: list[int] = []
        for index in range(committed, position):
            token = pending.get(index)
            if token is None:
                raise RuntimeError(
                    f"no token for {state.request.request_id} at position "
                    f"{index} while sampling position {position}"
                )
            values.append(token)
        return tuple(values)

    def _pending_device_outputs(
        self, state: object, position: int
    ) -> tuple[torch.Tensor, ...]:
        """Uncommitted completion history as device scalars, in position order."""
        committed = len(state.outputs)
        if position <= committed:
            return ()
        pending = self._future_device_tokens.get(state.request.request_id, {})
        values: list[torch.Tensor] = []
        for index in range(committed, position):
            token = pending.get(index)
            if token is None:
                raise RuntimeError(
                    f"no device token for {state.request.request_id} at position "
                    f"{index} while sampling position {position}"
                )
            values.append(token)
        return tuple(values)

    def _sample_device(
        self, state: object, logits: torch.Tensor, position: int
    ) -> _PendingDeviceToken:
        self._require_sampling_owner()
        if self._sampler is None:
            sample = DeviceSample(torch.argmax(logits).to(dtype=torch.int64))
        else:
            sample = self._sampler.sample_device(
                state.request.sampling_identity,
                state.request.sampling,
                position,
                logits,
                prompt=state.request.prompt_token_ids,
                outputs=state.outputs,
                pending_outputs=self._pending_device_outputs(state, position),
                history_epoch=getattr(state, "output_epoch", 0),
                eos_token_id=state.request.eos_token_id,
            )
        request_id = state.request.request_id
        self._remember_device(request_id, position, sample.token_id)
        return _PendingDeviceToken(
            sample=sample,
            on_resolve=lambda token: self._remember(request_id, position, token),
        )

    def _can_sample_device(self, state: object, logits: torch.Tensor) -> bool:
        if logits.device.type != "cuda":
            return False
        sampling = getattr(state.request, "sampling", None)
        return sampling is None or not sampling.needs_grammar

    def _sample_record(
        self, state: object, logits: torch.Tensor, position: int
    ) -> SampledToken | _PendingDeviceToken:
        if self._can_sample_device(state, logits):
            return self._sample_device(state, logits, position)
        return self._sample(state, logits, position)

    def _sample_rows(
        self,
        chunks: list[ScheduledChunk],
        states: Mapping[str, object],
        logits: torch.Tensor,
    ) -> tuple[SampledToken | _PendingDeviceToken, ...]:
        """Select a decode batch without copying every vocab row to the host.

        The common serving request is pure greedy with no logprob report. All
        rows can then argmax on-device.  Their ids patch the next decode inputs
        device-to-device; one deferred D2H carries the public StepOutput later.
        """
        self._require_sampling_owner()
        if logits.device.type == "cuda" and all(
            self._can_sample_device(states[chunk.request_id], logits[index])
            for index, chunk in enumerate(chunks)
        ):
            direct = self._sampler is None
            if self._sampler is not None:
                direct = all(
                    self._sampler.can_argmax_logits(
                        states[chunk.request_id].request.sampling_identity,
                        states[chunk.request_id].request.sampling,
                        states[chunk.request_id].request.eos_token_id,
                    )
                    for chunk in chunks
                )
            if direct:
                token_ids = torch.argmax(logits, dim=-1).to(dtype=torch.int64)
                records: list[_PendingDeviceToken] = []
                for index, chunk in enumerate(chunks):
                    sample = DeviceSample(token_ids[index])
                    self._remember_device(
                        chunk.request_id, chunk.position, sample.token_id
                    )
                    records.append(
                        _PendingDeviceToken(
                            sample=sample,
                            on_resolve=lambda token, request_id=chunk.request_id,
                            position=chunk.position: self._remember(
                                request_id, position, token
                            ),
                        )
                    )
                return tuple(records)
            return tuple(
                self._sample_device(
                    states[chunk.request_id],
                    logits[index],
                    position=chunk.position,
                )
                for index, chunk in enumerate(chunks)
            )

        direct = self._sampler is None
        if self._sampler is not None:
            direct = all(
                self._sampler.can_argmax_logits(
                    states[chunk.request_id].request.sampling_identity,
                    states[chunk.request_id].request.sampling,
                    states[chunk.request_id].request.eos_token_id,
                )
                for chunk in chunks
            )
        if direct:
            token_ids = torch.argmax(logits, dim=-1).to(device="cpu").tolist()
            return tuple(SampledToken(int(token_id)) for token_id in token_ids)
        return tuple(
            self._sample_record(
                states[chunk.request_id],
                logits[index],
                position=chunk.position,
            )
            for index, chunk in enumerate(chunks)
        )

    def _effective_outputs(self, state: object, position: int) -> tuple[int, ...]:
        """Committed outputs, extended with any in-flight token before ``position``.

        Under overlap the snapshot for step N+1 is taken before step N commits,
        so `state.outputs` is short by exactly the tokens this runner has already
        sampled. Sampling reads that history for the penalties, so it needs the
        same view the decode input does.
        """
        outputs = tuple(state.outputs)
        if position <= len(outputs):
            return outputs
        pending = self._future_tokens.get(state.request.request_id, {})
        extended = list(outputs)
        for index in range(len(outputs), position):
            if index not in pending:
                # a gap means the history is genuinely unknown; penalties over a
                # silently-short history are worse than a loud failure
                raise RuntimeError(
                    f"no token for {state.request.request_id} at position {index} "
                    f"while sampling position {position}"
                )
            extended.append(pending[index])
        return tuple(extended)

    def execute(
        self, scheduled: tuple[ScheduledChunk, ...], states: Mapping[str, object]
    ) -> Mapping[str, tuple[SampledToken, ...]]:
        return self._execute(scheduled, states, sample=True)

    def execute_passive(
        self, scheduled: tuple[ScheduledChunk, ...], states: Mapping[str, object]
    ) -> Mapping[str, tuple[SampledToken, ...]]:
        """Run this TP shard's model/KV work without owning sampling.

        Non-zero TP ranks consume rank 0's canonical device-token packet after
        this method returns.  They therefore must not advance RNG/grammar state,
        build logprobs, or enqueue a StepOutput D2H copy of their own.
        """
        return self._execute(scheduled, states, sample=False)

    def _execute(
        self,
        scheduled: tuple[ScheduledChunk, ...],
        states: Mapping[str, object],
        *,
        sample: bool,
    ) -> Mapping[str, tuple[SampledToken, ...]]:
        self._reap_deferred_outputs()
        sampled: (
            dict[str, tuple[SampledToken | _PendingDeviceToken, ...]] | None
        ) = {} if sample else None
        prefills = [chunk for chunk in scheduled if chunk.is_prefill]
        decodes = [chunk for chunk in scheduled if not chunk.is_prefill]
        if (
            len(prefills) >= 2
            and self._batched_prefill_enabled
            and self._prefill_batch_gap is None
        ):
            self._execute_prefill_batch(prefills, states, sampled)
        else:
            for chunk in prefills:
                self._execute_prefill(
                    chunk, states[chunk.request_id], sampled
                )
        # C4: single-token decodes for all sequences run as ONE tensor forward
        # (byte-identical to per-sequence decode; see test_batched_decode).
        # Eager execution keeps the cheaper per-sequence path for B=1, but graph
        # mode must use its tensor/static-buffer path at every supported bucket:
        # single-stream launch overhead is one of CUDA graph's primary targets.
        if decodes and (
            self._graph is not None
            or len(decodes) >= 2
            or (self._device.type == "cuda" and self._tensor_decode_supported)
        ):
            self._execute_decode_batch(decodes, states, sampled)
        else:
            for chunk in decodes:
                self._execute_decode(chunk, states[chunk.request_id], sampled)
        if sampled is None:
            return {}
        if any(
            isinstance(token, _PendingDeviceToken)
            for tokens in sampled.values()
            for token in tokens
        ):
            output = _DeferredStepOutput(sampled, self._output_copy_stream)
            self._deferred_outputs.append(output)
            return output
        return {
            request_id: tuple(
                token for token in tokens if isinstance(token, SampledToken)
            )
            for request_id, tokens in sampled.items()
        }

    @staticmethod
    def _chunk_emits_token(chunk: ScheduledChunk, state: object) -> bool:
        """Whether this scheduled slot produces one sampled token."""
        if not chunk.is_prefill:
            return True
        prompt = state.request.prompt_token_ids
        return state.prefill_done and state.computed_prompt == len(prompt)

    def make_sampling_token_packet(
        self,
        scheduled: tuple[ScheduledChunk, ...],
        states: Mapping[str, object],
        sampled: Mapping[str, tuple[SampledToken, ...]] | None = None,
    ) -> torch.Tensor:
        """Build the fixed-layout rank-0 token packet or a follower receive buffer.

        One int64 slot corresponds to one ``ScheduledChunk``.  Partial-prefill
        slots carry ``-1``; every emitting slot carries exactly one token id.
        Keeping the shape tied to the already-broadcast chunk tuple means all
        ranks enter the same tensor collective even for mixed prefill/decode
        batches, without request-id or Python-result traffic.
        """
        chunks = tuple(scheduled)
        packet = torch.full(
            (len(chunks),), -1, dtype=torch.int64, device=self._device
        )
        expected = [
            chunk.request_id
            for chunk in chunks
            if self._chunk_emits_token(chunk, states[chunk.request_id])
        ]
        if len(expected) != len(set(expected)):
            raise RuntimeError(
                "TP sampling packet cannot represent more than one emitted token "
                "for the same request in one model step"
            )
        if sampled is None:
            return packet

        actual = set(sampled)
        expected_set = set(expected)
        if actual != expected_set:
            missing = sorted(expected_set - actual)
            extra = sorted(actual - expected_set)
            raise RuntimeError(
                "TP sampling owner output does not match the scheduled token "
                f"layout: missing={missing}, extra={extra}"
            )
        records: Mapping[
            str, tuple[SampledToken | _PendingDeviceToken, ...]
        ]
        if isinstance(sampled, _DeferredStepOutput):
            records = sampled.raw_records()
        else:
            records = sampled
        for index, chunk in enumerate(chunks):
            state = states[chunk.request_id]
            if not self._chunk_emits_token(chunk, state):
                continue
            values = records[chunk.request_id]
            if len(values) != 1:
                raise RuntimeError(
                    "TP sampling owner must emit exactly one token for "
                    f"{chunk.request_id!r} at position {chunk.position}; "
                    f"got {len(values)}"
                )
            record = values[0]
            if isinstance(record, _PendingDeviceToken):
                packet[index].copy_(record.sample.token_id)
            elif isinstance(record, SampledToken):
                if record.token_id < 0:
                    raise RuntimeError(
                        f"TP sampling owner emitted invalid token id {record.token_id} "
                        f"for {chunk.request_id!r} at position {chunk.position}"
                    )
                packet[index] = record.token_id
            else:  # pragma: no cover - a malformed internal StepOutput
                raise TypeError(
                    "TP sampling owner returned an unsupported token record "
                    f"{type(record).__name__}"
                )
        return packet

    def adopt_sampling_token_packet(
        self,
        scheduled: tuple[ScheduledChunk, ...],
        states: Mapping[str, object],
        packet: torch.Tensor,
    ) -> None:
        """Adopt rank 0's device ids before this rank can execute another step."""
        chunks = tuple(scheduled)
        if packet.dtype != torch.int64 or packet.ndim != 1:
            raise RuntimeError(
                "TP sampling packet must be a one-dimensional int64 tensor; "
                f"got dtype={packet.dtype}, shape={tuple(packet.shape)}"
            )
        if packet.numel() != len(chunks):
            raise RuntimeError(
                "TP sampling packet length does not match the scheduled layout: "
                f"got {packet.numel()}, expected {len(chunks)}"
            )
        if packet.device != self._device:
            raise RuntimeError(
                "TP sampling packet is on the wrong device: "
                f"got {packet.device}, expected {self._device}"
            )
        for index, chunk in enumerate(chunks):
            state = states[chunk.request_id]
            emits = self._chunk_emits_token(chunk, state)
            if self._device.type == "cpu":
                value = int(packet[index])
                if emits and value < 0:
                    raise RuntimeError(
                        "TP sampling packet is missing the authoritative token for "
                        f"{chunk.request_id!r} at position {chunk.position}"
                    )
                if not emits and value != -1:
                    raise RuntimeError(
                        "TP sampling packet emitted an unexpected token for partial "
                        f"prefill {chunk.request_id!r} at position {chunk.position}"
                    )
            if emits:
                # Retain the packet-backed scalar itself: the next decode input
                # consumes it D2D on the same stream, with no host scalar sync.
                self._remember_device(
                    chunk.request_id, chunk.position, packet[index]
                )

    def _reap_deferred_outputs(self) -> None:
        while self._deferred_outputs and self._deferred_outputs[0].ready():
            self._deferred_outputs.popleft()

    def _execute_prefill(
        self, chunk: ScheduledChunk, state, sampled: dict | None
    ) -> None:
        prompt = state.request.prompt_token_ids
        page_table = list(state.allocation.pages) + list(state.decode_pages)
        cached = state.allocation.num_cached_tokens if state.allocation else 0
        end = state.computed_prompt
        start = end - chunk.num_tokens
        hidden = self._model.forward_tokens(
            torch.tensor(prompt[start:end], dtype=torch.long, device=self._device),
            torch.arange(start, end, device=self._device),
            self._pool, page_table, seq_len=end, write_from=cached,
        )
        self._prefill_rows_executed = (
            getattr(self, "_prefill_rows_executed", 0) + 1
        )
        self._prefill_model_calls = (
            getattr(self, "_prefill_model_calls", 0) + 1
        )
        self._prefill_sequential_rows = (
            getattr(self, "_prefill_sequential_rows", 0) + 1
        )
        if sampled is not None and state.prefill_done and end == len(prompt):
            logits = self._model.logits(hidden[-1])
            token = self._sample_record(state, logits, position=0)
            if isinstance(token, SampledToken):
                self._remember(chunk.request_id, 0, token)
            sampled[chunk.request_id] = (token,)

    def _execute_prefill_batch(
        self,
        chunks: list[ScheduledChunk],
        states: Mapping[str, object],
        sampled: dict | None,
    ) -> None:
        """Run compatible ScheduledChunks through one flat model chain."""
        rows: list[PrefillSequence] = []
        for chunk in chunks:
            state = states[chunk.request_id]
            allocation = state.allocation
            if allocation is None:
                raise RuntimeError(
                    f"prefill request {chunk.request_id!r} has no KV allocation"
                )
            prompt = state.request.prompt_token_ids
            end = state.computed_prompt
            start = end - chunk.num_tokens
            rows.append(
                PrefillSequence(
                    request_id=chunk.request_id,
                    token_ids=tuple(prompt[start:end]),
                    page_table=tuple(allocation.pages)
                    + tuple(state.decode_pages),
                    chunk_start=start,
                    seq_len=end,
                    # Do not clamp this to `start`: a full-cache hit recomputes
                    # the final query while retaining its shared KV slot.
                    write_from=allocation.num_cached_tokens,
                )
            )
        batch = build_prefill_batch(
            rows, page_size=self._pool.page_size, device=self._device
        )
        hidden = self._model.forward_prefill_batch(batch, self._pool)
        self._prefill_rows_executed = (
            getattr(self, "_prefill_rows_executed", 0) + len(chunks)
        )
        self._prefill_model_calls = (
            getattr(self, "_prefill_model_calls", 0) + 1
        )
        self._prefill_batched_groups = (
            getattr(self, "_prefill_batched_groups", 0) + 1
        )
        if sampled is None:
            return

        emitting_chunks: list[ScheduledChunk] = []
        terminal_rows: list[int] = []
        for index, chunk in enumerate(chunks):
            state = states[chunk.request_id]
            if (
                state.prefill_done
                and batch.seq_lens[index]
                == len(state.request.prompt_token_ids)
            ):
                emitting_chunks.append(chunk)
                terminal_rows.append(batch.qo_indptr[index + 1] - 1)
        if not emitting_chunks:
            return

        selected = torch.tensor(
            terminal_rows, dtype=torch.long, device=hidden.device
        )
        logits = self._model.logits(hidden.index_select(0, selected))
        records = self._sample_rows(emitting_chunks, states, logits)
        for chunk, token in zip(emitting_chunks, records, strict=True):
            if isinstance(token, SampledToken):
                self._remember(chunk.request_id, 0, token)
            sampled[chunk.request_id] = (token,)

    def _remember(self, request_id: str, position: int, token: SampledToken) -> None:
        """Keep this step's token so later steps can read it before it commits.

        A decode input needs `position - 1`; the sampler's penalties need every
        uncommitted token before `position`. Both are served from here.

        Trimming is driven by what has been COMMITTED, not by a fixed depth.
        `OverlapEngineCore` accepts any `pipeline_depth >= 1`, so a constant cap
        would silently drop a position a deeper pipeline still needs — and the
        symptom would be a RuntimeError mid-run, not a wrong answer, but only
        because the lookup fails loudly. Anything at or below the committed
        length is redundant: `_effective_outputs` and `_previous_token` both
        prefer committed values there.
        """
        pending = self._future_tokens.setdefault(request_id, {})
        pending[position] = token.token_id

    def _remember_device(
        self, request_id: str, position: int, token: torch.Tensor
    ) -> None:
        pending = self._future_device_tokens.setdefault(request_id, {})
        pending[position] = token

    def _forget_committed(self, request_id: str, committed: int) -> None:
        """Drop in-flight tokens the scheduler has since committed."""
        pending = self._future_tokens.get(request_id)
        if pending:
            for position in [key for key in pending if key < committed]:
                del pending[position]
        device_pending = self._future_device_tokens.get(request_id)
        if device_pending:
            for position in [key for key in device_pending if key < committed - 1]:
                del device_pending[position]

    def _previous_token(self, state, position: int) -> int | torch.Tensor:
        """The token at ``position - 1``, committed or still in flight.

        Committed outputs win: after `update()` they are the authority, and a
        speculative rollback replaces in-flight values that the future buffer
        would otherwise still hold.
        """
        index = position - 1
        outputs = state.outputs
        # Speculative target scoring deliberately overlays the scheduler's
        # committed completion with a draft prefix.  That explicit overlay must
        # beat a target token retained from the preceding score; normal overlap
        # snapshots keep the device-first fast path below.
        if getattr(state, "outputs_override", False) and index < len(outputs):
            return outputs[index]
        device_pending = self._future_device_tokens.get(
            state.request.request_id, {}
        )
        if index in device_pending:
            return device_pending[index]
        if index < len(outputs):
            return outputs[index]
        pending = self._future_tokens.get(state.request.request_id, {})
        if index in pending:
            return pending[index]
        raise RuntimeError(
            f"no token for {state.request.request_id} at position {index}: "
            f"{len(outputs)} committed, in-flight {sorted(pending)}"
        )

    def _decode_inputs(self, chunk: ScheduledChunk, state):
        prompt = state.request.prompt_token_ids
        position = chunk.position
        # whatever the scheduler has committed is authoritative from here on, so
        # the in-flight copies of it are dead weight
        input_token = self._previous_token(state, position) if position > 0 else prompt[-1]
        self._forget_committed(state.request.request_id, len(state.outputs))
        absolute = len(prompt) + position - 1
        cached = state.allocation.num_cached_tokens if state.allocation else 0
        page_table = list(state.allocation.pages) + list(state.decode_pages)
        return input_token, absolute, page_table, cached

    def _allocate_decode_slots(self, size: int) -> None:
        """(Re)allocate the persistent slots, doubling so growth is amortized."""
        current = 0 if self._decode_slots is None else self._decode_slots.numel()
        capacity = max(8, size, 2 * current)
        self._decode_slots = torch.zeros(capacity, dtype=torch.long, device=self._device)
        self._decode_positions = torch.zeros(capacity, dtype=torch.long, device=self._device)

    def _acquire_slot_staging(
        self, size: int
    ) -> tuple[torch.Tensor, torch.cuda.Event]:
        """Return an idle pinned buffer without waiting for the device.

        Overlap may enqueue several steps before the oldest output commits.  A
        single staging row therefore needs an event wait before reuse, which was
        the last explicit synchronization in the decode feedback path.  This
        small pool grows only to the number of genuinely in-flight H2D copies
        and reuses completed rows through non-blocking ``query()``.
        """
        for index, (staging, event) in enumerate(self._slot_staging_pool):
            if not event.query():
                continue
            if staging.shape[1] < size:
                capacity = max(size, 2 * staging.shape[1])
                staging = torch.zeros(2, capacity, dtype=torch.long).pin_memory()
                event = torch.cuda.Event()
                self._slot_staging_pool[index] = (staging, event)
            self._slot_staging = staging
            self._slot_copy_done = event
            return staging, event
        capacity = max(8, size)
        staging = torch.zeros(2, capacity, dtype=torch.long).pin_memory()
        event = torch.cuda.Event()
        self._slot_staging_pool.append((staging, event))
        self._slot_staging = staging
        self._slot_copy_done = event
        return staging, event

    def _decode_input_slots(
        self, tokens: list[int | torch.Tensor], positions: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write this step's decode inputs INTO the persistent device slots.

        m2 §5 names this: "the GPU runner owns a future-token device buffer
        indexed by (request, position)". The slots are allocated once and patched
        in place, so no decode step allocates a device tensor and BOTH decode
        paths — batched and single-request — read the same memory. Views are
        returned, so a caller that kept last step's view sees this step's values;
        that aliasing is the property, and `test_decode_input_slots.py` pins it.

        CUDA samples arrive as device scalar tensors and are copied directly
        into their row. Host integers remain supported for CPU/structured
        compatibility; only those rows use the pinned batched H2D staging pool.
        """
        size = len(tokens)
        if self._decode_slots is None or self._decode_slots.numel() < size:
            self._allocate_decode_slots(size)
        assert self._decode_slots is not None and self._decode_positions is not None
        host_tokens = [
            0 if isinstance(token, torch.Tensor) else token for token in tokens
        ]
        if self._device.type == "cuda":
            staging, copy_done = self._acquire_slot_staging(size)
            staging[1, :size].copy_(torch.as_tensor(positions, dtype=torch.long))
            self._decode_positions[:size].copy_(
                staging[1, :size], non_blocking=True
            )
            if any(not isinstance(token, torch.Tensor) for token in tokens):
                staging[0, :size].copy_(
                    torch.as_tensor(host_tokens, dtype=torch.long)
                )
                self._decode_slots[:size].copy_(
                    staging[0, :size], non_blocking=True
                )
        else:
            self._decode_slots[:size].copy_(
                torch.as_tensor(host_tokens, dtype=torch.long)
            )
            self._decode_positions[:size].copy_(
                torch.as_tensor(positions, dtype=torch.long)
            )
        for index, token in enumerate(tokens):
            if isinstance(token, torch.Tensor):
                self._decode_slots[index].copy_(
                    token.to(device=self._device, dtype=torch.long)
                )
        if self._device.type == "cuda":
            copy_done.record()
        return self._decode_slots[:size], self._decode_positions[:size]

    def _execute_decode(
        self, chunk: ScheduledChunk, state, sampled: dict | None
    ) -> None:
        input_token, absolute, page_table, cached = self._decode_inputs(chunk, state)
        # the single-request path uses the SAME slots as the batched one: a
        # workload that drops to one request must not fall back to rebuilding a
        # fresh device tensor every step
        token_slot, position_slot = self._decode_input_slots([input_token], [absolute])
        hidden = self._model.forward_tokens(
            token_slot, position_slot,
            self._pool, page_table, seq_len=absolute + 1, write_from=cached,
        )
        if sampled is None:
            return
        logits = self._model.logits(hidden[-1])
        token = self._sample_record(state, logits, position=chunk.position)
        if isinstance(token, SampledToken):
            self._remember(chunk.request_id, chunk.position, token)
        sampled[chunk.request_id] = (token,)

    def _graph_logits(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        page_tables: list[list[int]],
        seq_lens: list[int],
        write_from: list[int] | None = None,
    ) -> torch.Tensor:
        """Route this decode through capture/replay (m17 D1).

        `build_decode_batch` pads the ragged page lists into the [B, max_pages]
        tensor the captured kernels index; `GraphStepExecutor` falls back to
        eager for an oversize batch or a page table wider than the static buffer, so
        this cannot fail on shape.

        The width padding uses the SAME reserved scratch page as the row padding:
        a short row's tail entries are only ever read (and then masked off by
        seq_lens), but pointing them at an allocatable page leaves the class of
        bug [P1] found one indexing mistake away.
        """
        from kairyu.engine.core.step_executor import build_decode_batch

        assert self._graph_scratch_page is not None  # set with self._graph
        if write_from is None:
            write_from = [0] * len(tokens)
        batch = build_decode_batch(
            token_ids=tokens,
            positions=positions,
            page_lists=page_tables,
            seq_lens=seq_lens,
            max_pages=max(len(table) for table in page_tables),
            scratch_page=self._graph_scratch_page,
            write_from=write_from,
            device=self._device,
        )
        return self._graph.execute_decode(batch)

    def _eager_tensor_logits(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        page_tables: list[list[int]],
        seq_lens: list[int],
        write_from: list[int],
    ) -> torch.Tensor:
        """One tensor-metadata eager forward, with no per-row device reads.

        Ragged page-table tails repeat each request's last owned page. They are
        masked by seq_lens and there are no synthetic rows, so eager execution
        needs no graph scratch-page reservation.
        """
        from kairyu.engine.core.step_executor import build_decode_batch

        batch = build_decode_batch(
            token_ids=tokens,
            positions=positions,
            page_lists=page_tables,
            seq_lens=seq_lens,
            max_pages=max(len(table) for table in page_tables),
            scratch_page=None,
            write_from=write_from,
            device=self._device,
        )
        hidden = self._eager_tensor_hidden(batch)
        return self._model.logits(hidden)

    def _eager_tensor_hidden(self, batch) -> torch.Tensor:
        """Run eager tensor decode through KV write, without forcing lm_head."""
        self._model.plan_decode_tensors(
            self._pool, batch.page_tables, batch.seq_lens
        )
        return self._model.forward_decode_tensors(
            batch.token_ids,
            batch.positions,
            self._pool,
            batch.page_tables,
            batch.seq_lens,
            batch.write_from,
        )

    def _execute_decode_batch(
        self,
        chunks: list[ScheduledChunk],
        states: Mapping[str, object],
        sampled: dict | None,
    ) -> None:
        tokens, positions, page_tables, seq_lens, write_from = [], [], [], [], []
        for chunk in chunks:
            input_token, absolute, page_table, cached = self._decode_inputs(
                chunk, states[chunk.request_id]
            )
            tokens.append(input_token)
            positions.append(absolute)
            page_tables.append(page_table)
            seq_lens.append(absolute + 1)
            write_from.append(cached)
        # Every branch consumes the same persistent token/position slots. The
        # tensor path must not regress #143 by rebuilding those inputs while it
        # tensorizes page-table metadata.
        token_slots, position_slots = self._decode_input_slots(tokens, positions)
        if self._graph is not None:
            logits = self._graph_logits(
                token_slots, position_slots, page_tables, seq_lens, write_from
            )
        elif self._tensor_decode_supported:
            from kairyu.engine.core.step_executor import build_decode_batch

            batch = build_decode_batch(
                token_ids=token_slots,
                positions=position_slots,
                page_lists=page_tables,
                seq_lens=seq_lens,
                max_pages=max(len(table) for table in page_tables),
                scratch_page=None,
                write_from=write_from,
                device=self._device,
            )
            hidden = self._eager_tensor_hidden(batch)
            logits = self._model.logits(hidden) if sampled is not None else None
        else:
            hidden = self._model.forward_decode_batch(
                token_slots, position_slots,
                self._pool, page_tables, seq_lens, write_from,
                position_values=positions,
            )
            logits = (
                self._model.logits(hidden) if sampled is not None else None
            )  # [B, vocab]
        if sampled is None:
            return
        assert logits is not None
        tokens = self._sample_rows(chunks, states, logits)
        for chunk, token in zip(chunks, tokens, strict=True):
            if isinstance(token, SampledToken):
                self._remember(chunk.request_id, chunk.position, token)
            sampled[chunk.request_id] = (token,)
