"""Kairyu engine core exposed as an EngineBackend (design m2 §1, backend name "kairyu").

Full-stack integration on CPU: OpenAI server / LLM API → this backend →
``EngineLoop`` (tokenizer + Scheduler + RadixKVCache + runner). The model
forward is a deterministic toy runner on CPU; the GPU phase swaps in the real
ModelRunner behind the same protocol — nothing above it changes.

Threading discipline (m8 D1): all scheduler mutations happen inside
``EngineLoop.step()`` on the step thread; the event loop only enqueues ops and
reads queues. The ZMQ process-split backend ("kairyu-proc", m8 D6) drives the
same ``EngineLoop`` from a child process.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Mapping
from pathlib import Path

from kairyu.engine.backend import (
    EngineReadiness,
    GenerationRequest,
    GenerationResult,
    GenerationUsage,
    prompt_with_tool_intent,
)
from kairyu.engine.core.comm import FakeCommunicator
from kairyu.engine.core.pd_loop import PDLoopAdapter
from kairyu.engine.core.radix_kv import RadixKVCache
from kairyu.engine.core.sampling_types import SampledToken, mix_seed
from kairyu.engine.core.scheduler import ScheduledChunk, Scheduler
from kairyu.engine.core.spec_runner import SpeculativeRunner
from kairyu.engine.core.tp_runner import TPModelRunner, validate_tp_degree
from kairyu.engine.engine_loop import EngineLoop, StreamUpdate
from kairyu.engine.registry import register_backend
from kairyu.engine.tokenizer import (
    Tokenizer,
    grammar_vocabulary,
    resolve_tokenizer,
)
from kairyu.outputs import CompletionOutput

_VOCAB_SIZE = 50_000
_DECODE_MODES = frozenset({"eager", "cuda_graph"})
logger = logging.getLogger(__name__)


class _ToyRunner:
    """Deterministic CPU stand-in for the GPU model forward (greedy only —
    sampling params take effect with a Sampler-equipped runner, m8 D2)."""

    def __init__(self, *, sampling_owner: bool = True) -> None:
        self._sampling_owner = sampling_owner
        self._future_tokens: dict[str, dict[int, int]] = {}

    def execute(
        self, scheduled: tuple[ScheduledChunk, ...], states: Mapping[str, object]
    ) -> dict[str, tuple[SampledToken, ...]]:
        if not self._sampling_owner:
            raise RuntimeError(
                "only rank 0 may sample in tensor-parallel execution; "
                "followers must use execute_passive()"
            )
        sampled: dict[str, tuple[SampledToken, ...]] = {}
        for chunk in scheduled:
            state = states[chunk.request_id]
            if self._chunk_emits_token(chunk, state):
                seed = sum(state.request.prompt_token_ids) if state.request.prompt_token_ids else 0
                token_id = (seed + 31 * chunk.position) % _VOCAB_SIZE
                sampled[chunk.request_id] = (SampledToken(token_id),)
        return sampled

    def execute_passive(
        self, scheduled: tuple[ScheduledChunk, ...], states: Mapping[str, object]
    ) -> dict[str, tuple[SampledToken, ...]]:
        """Toy has no model/KV work, but followers still enter the passive seam."""
        return {}

    @staticmethod
    def _chunk_emits_token(chunk: ScheduledChunk, state: object) -> bool:
        return not chunk.is_prefill or state.prefill_done

    def make_sampling_token_packet(
        self,
        scheduled: tuple[ScheduledChunk, ...],
        states: Mapping[str, object],
        sampled: Mapping[str, tuple[SampledToken, ...]] | None = None,
    ):
        """Build the same one-int64-per-chunk layout as ``PagedModelRunner``."""
        import torch

        chunks = tuple(scheduled)
        packet = torch.full((len(chunks),), -1, dtype=torch.int64)
        expected = {
            chunk.request_id
            for chunk in chunks
            if self._chunk_emits_token(chunk, states[chunk.request_id])
        }
        if sampled is None:
            return packet
        actual = set(sampled)
        if actual != expected:
            raise RuntimeError(
                "TP sampling owner output does not match the scheduled token "
                f"layout: missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )
        for index, chunk in enumerate(chunks):
            if chunk.request_id not in expected:
                continue
            records = sampled[chunk.request_id]
            if len(records) != 1 or records[0].token_id < 0:
                raise RuntimeError(
                    "TP sampling owner must emit exactly one valid token for "
                    f"{chunk.request_id!r} at position {chunk.position}"
                )
            packet[index] = records[0].token_id
        return packet

    def adopt_sampling_token_packet(
        self,
        scheduled: tuple[ScheduledChunk, ...],
        states: Mapping[str, object],
        packet,
    ) -> None:
        """Overwrite any rank-local future ids with rank 0's canonical packet."""
        import torch

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
        for index, chunk in enumerate(chunks):
            emits = self._chunk_emits_token(chunk, states[chunk.request_id])
            token_id = int(packet[index])
            if emits and token_id < 0:
                raise RuntimeError(
                    "TP sampling packet is missing the authoritative token for "
                    f"{chunk.request_id!r} at position {chunk.position}"
                )
            if not emits and token_id != -1:
                raise RuntimeError(
                    "TP sampling packet emitted an unexpected token for partial "
                    f"prefill {chunk.request_id!r} at position {chunk.position}"
                )
            if emits:
                self._future_tokens.setdefault(chunk.request_id, {})[chunk.position] = token_id

    def release(self, request_id: str) -> None:
        self._future_tokens.pop(request_id, None)


def build_engine_loop(
    *,
    num_pages: int = 4096,
    page_size: int = 16,
    max_num_batched_tokens: int = 2048,
    max_num_seqs: int = 256,
    priority_age_s: float | None = 60.0,
    runner: object | None = None,
    tensor_parallel_size: int = 1,
    tokenizer: str | Tokenizer | None = None,
    speculative: str | None = None,
    speculative_tokens: int = 4,
    model_path: str | None = None,
    pd_separation: bool = False,
    pipeline_depth: int = 1,
    decode_mode: str = "eager",
    cuda_graph_max_batch: int = 8,
    cuda_graph_max_pages: int = 512,
    cuda_graph_warmup_iters: int = 3,
) -> tuple[EngineLoop, RadixKVCache, Scheduler | PDLoopAdapter]:
    """Assemble the engine stack; shared by KairyuBackend and the ZMQ service.

    ``model_path`` loads a real checkpoint (m12 D5): DenseDecoder +
    PagedKVPool + PagedModelRunner + Sampler, tokenizer from the same dir
    unless overridden. Mutually exclusive with ``runner``. Real-model TP > 1
    spawns the ``DistTPLauncher`` group (rank 0 here, ranks 1.. as workers) and
    drives it via ``DistTPModelRunner``; the loop's ``.tp_launcher`` handle must
    be ``shutdown()`` on serve teardown. A caller-supplied ``runner`` is a
    single rank-local object and therefore requires ``tensor_parallel_size=1``.
    """
    if pipeline_depth < 1:
        raise ValueError(f"pipeline_depth must be >= 1, got {pipeline_depth}")
    if speculative is not None and speculative != "ngram":
        raise ValueError(f"unknown speculative mode {speculative!r} (only 'ngram')")
    if decode_mode not in _DECODE_MODES:
        known = ", ".join(sorted(_DECODE_MODES))
        raise ValueError(f"unknown decode_mode {decode_mode!r}; choose one of: {known}")
    if runner is not None and model_path is None and tensor_parallel_size > 1:
        raise ValueError(
            "custom runner with tensor_parallel_size > 1 is unsupported: "
            "tensor parallelism needs one distinct rank-local runner per rank; "
            "use model_path for real TP or set tensor_parallel_size=1"
        )
    graph_decode = decode_mode == "cuda_graph"
    if graph_decode:
        if model_path is None:
            raise ValueError("decode_mode='cuda_graph' needs a real model_path")
        if cuda_graph_max_batch < 1 or cuda_graph_max_pages < 1:
            raise ValueError("cuda_graph_max_batch and cuda_graph_max_pages must be >= 1")
        if cuda_graph_warmup_iters < 0:
            raise ValueError("cuda_graph_warmup_iters must be >= 0")
        if num_pages < 2:
            raise ValueError("CUDA graph decode needs at least 2 KV pages")
        if cuda_graph_max_pages >= num_pages:
            raise ValueError(
                f"cuda_graph_max_pages={cuda_graph_max_pages} must be smaller "
                f"than num_pages={num_pages} so one page can remain scratch"
            )
    if model_path is not None and runner is not None:
        raise ValueError("model_path and runner are mutually exclusive")
    if pd_separation:
        if model_path is None:
            raise ValueError("pd_separation needs a model_path")
        if tensor_parallel_size > 1:
            # the coordinator owns two engines; sharding each of them is m6
            # stage 5.3 territory, not something to half-do here
            raise ValueError("pd_separation with tensor_parallel_size > 1 is not supported")
        if speculative is not None:
            raise ValueError("pd_separation with speculative decoding is not supported")
        if graph_decode:
            raise ValueError("pd_separation with CUDA graph decode is not supported")
        return _build_pd_loop(
            model_path=model_path,
            num_pages=num_pages,
            page_size=page_size,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            priority_age_s=priority_age_s,
            tokenizer=tokenizer,
            pipeline_depth=pipeline_depth,
        )

    if model_path is not None and tensor_parallel_size > 1:
        return _build_dist_tp_loop(
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            num_pages=num_pages,
            page_size=page_size,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            priority_age_s=priority_age_s,
            tokenizer=tokenizer,
            speculative=speculative,
            speculative_tokens=speculative_tokens,
            pipeline_depth=pipeline_depth,
            graph_decode=graph_decode,
            graph_max_batch=cuda_graph_max_batch,
            graph_max_pages=cuda_graph_max_pages,
            graph_warmup_iters=cuda_graph_warmup_iters,
        )
    if speculative is not None and tensor_parallel_size > 1:
        # The FakeCommunicator TP path has rank-local toy runners but no
        # rank-local speculative scorers. The real multi-process path above
        # constructs the complete stack independently on every rank.
        raise ValueError(
            "speculative decoding with tensor_parallel_size > 1 requires a real "
            "model (model_path); the in-process toy TP path has no rank-local "
            "speculative scorers"
        )

    default_eos: int | None = None
    default_stop_ids: tuple[int, ...] = ()
    num_kv_heads_for_tp = None
    if model_path is not None:
        import torch

        from kairyu.engine.core.attention import select_backend
        from kairyu.engine.core.hw_profile import probe
        from kairyu.engine.core.kv_pool import PagedKVPool
        from kairyu.engine.core.model_runner import PagedModelRunner
        from kairyu.engine.core.sampler import Sampler
        from kairyu.models.loader import load_model

        # deploy day is config-free: the probed profile picks the kernel AND the
        # compute placement. GPU runs bf16 on-device (what the FlashInfer / FA2
        # kernels require — fp32 has no such kernels); CPU keeps fp32 on host, so
        # every CPU test path is byte-for-byte unchanged.
        profile = probe()
        gpu = profile.arch == "cuda"
        if graph_decode and not gpu:
            raise RuntimeError("decode_mode='cuda_graph' requires CUDA hardware")
        compute_device = "cuda" if gpu else "cpu"
        compute_dtype = torch.bfloat16 if gpu else torch.float32
        model, model_config, generation = load_model(
            model_path, dtype=compute_dtype, attention_backend=select_backend(profile)
        )
        model = model.to(compute_device)
        default_eos = generation.eos_token_id
        default_stop_ids = generation.stop_token_ids
        num_kv_heads_for_tp = model_config.num_key_value_heads
        resolved = resolve_tokenizer(tokenizer if tokenizer is not None else model_path)
        vocab_size = len(resolved.vocab())
        if vocab_size > model_config.vocab_size:
            raise ValueError(
                f"tokenizer vocab ({vocab_size}) exceeds the model's vocab_size "
                f"({model_config.vocab_size})"
            )
        grammar_vocab = grammar_vocabulary(resolved, model_vocab_size=model_config.vocab_size)
    else:
        resolved = resolve_tokenizer(tokenizer if tokenizer is not None else "toy")

    validate_tp_degree(
        tensor_parallel_size,
        **({"num_kv_heads": num_kv_heads_for_tp} if num_kv_heads_for_tp else {}),
    )
    cache = RadixKVCache(num_pages=num_pages, page_size=page_size)
    scheduler = Scheduler(
        cache,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        page_size=page_size,
        speculative_tokens=speculative_tokens if speculative else 0,
        priority_age_s=priority_age_s,
    )
    if model_path is not None:
        pool = PagedKVPool.for_cache(
            cache, model_config, dtype=compute_dtype, device=compute_device
        )
        graph_options = {}
        if graph_decode:
            from kairyu.engine.core.cuda_graph_gpu import CudaGraphBackend

            graph_options = {
                "graph_backend": CudaGraphBackend(warmup_iters=cuda_graph_warmup_iters),
                "graph_max_batch": cuda_graph_max_batch,
                "graph_max_pages": cuda_graph_max_pages,
            }
        runner = PagedModelRunner(
            model,
            pool,
            sampler=Sampler(vocab_provider=lambda: grammar_vocab),
            cache=cache,
            **graph_options,
        )
    if tensor_parallel_size > 1:
        # CPU-testable TP path (design m5 D1/D3): rank 0 is the only sampling
        # owner; followers enter the passive seam and adopt its fixed packet.
        active: object = TPModelRunner(
            rank_runners=tuple(
                _ToyRunner(sampling_owner=rank == 0) for rank in range(tensor_parallel_size)
            ),
            comms=FakeCommunicator.create_group(tensor_parallel_size),
        )
    else:
        active = runner or _ToyRunner()
    if speculative == "ngram":
        active = SpeculativeRunner(active)
    loop = EngineLoop(
        resolved,
        scheduler,
        active,
        default_eos_token_id=default_eos,
        default_stop_token_ids=default_stop_ids,
        pipeline_depth=pipeline_depth,
    )
    loop.tp_launcher = None  # single-process: nothing to tear down
    return loop, cache, scheduler


def _build_pd_loop(
    *,
    model_path: str,
    num_pages: int,
    page_size: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    priority_age_s: float | None,
    tokenizer: str | Tokenizer | None,
    pipeline_depth: int,
) -> tuple[EngineLoop, RadixKVCache, PDLoopAdapter]:
    """Serve through a prefill/decode pair (m18 D3, G2 stage 5.3).

    `PDCoordinator` had no serving entry point: `pd_factory` builds it, but a
    deployment could only reach it from Python. This is the `backend: kairyu`
    option that turns it on, so `pd_separation: true` in a deployment YAML is a
    real topology rather than a config surface m2 §2.4 reserved and never wired.

    The coordinator owns two schedulers and two runners; `PDLoopAdapter` is what
    makes that one scheduler and one runner from the loop's side — submissions
    enter at prefill, each step transfers the KV before planning decode. Handing
    the loop a bare coordinator would admit requests straight into the decode
    scheduler, with no prompt KV and no `execute` to call.

    The returned cache and scheduler are the ones the loop actually drives (the
    `build_engine_loop` contract): decode's cache, and the adapter itself.
    """
    from kairyu.engine.core.pd_factory import build_pd_coordinator
    from kairyu.models.loader import load_generation_defaults

    resolved = resolve_tokenizer(tokenizer if tokenizer is not None else model_path)
    coordinator = build_pd_coordinator(
        model_path=model_path,
        num_pages=num_pages,
        page_size=page_size,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        priority_age_s=priority_age_s,
        tokenizer=resolved,
    )
    generation = load_generation_defaults(model_path)
    adapter = PDLoopAdapter(coordinator)
    loop = EngineLoop(
        resolved,
        adapter,
        adapter,
        default_eos_token_id=generation.eos_token_id,
        default_stop_token_ids=generation.stop_token_ids,
        pipeline_depth=pipeline_depth,
    )
    loop.tp_launcher = None
    loop.pd_coordinator = coordinator
    return loop, adapter.kv_cache, adapter


def _build_dist_tp_loop(
    *,
    model_path: str,
    tensor_parallel_size: int,
    num_pages: int,
    page_size: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    priority_age_s: float | None,
    tokenizer: str | Tokenizer | None,
    speculative: str | None = None,
    speculative_tokens: int = 4,
    pipeline_depth: int = 1,
    graph_decode: bool = False,
    graph_max_batch: int = 8,
    graph_max_pages: int = 512,
    graph_warmup_iters: int = 3,
) -> tuple[EngineLoop, RadixKVCache, Scheduler]:
    """Real multi-process TP for `kairyu serve --tp N`: spawn the worker ranks,
    drive them through DistTPModelRunner, and expose the launcher on the loop so
    serve teardown can stop the workers cleanly."""
    from kairyu.engine.core.worker import DistTPLauncher
    from kairyu.models.loader import load_generation_defaults

    resolved = resolve_tokenizer(tokenizer if tokenizer is not None else model_path)
    raw_config = json.loads((Path(model_path) / "config.json").read_text())
    model_vocab_size = int(raw_config["vocab_size"])
    grammar_vocab = grammar_vocabulary(resolved, model_vocab_size=model_vocab_size)
    cache = RadixKVCache(num_pages=num_pages, page_size=page_size)
    scheduler = Scheduler(
        cache,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        page_size=page_size,
        speculative_tokens=speculative_tokens if speculative else 0,
        priority_age_s=priority_age_s,
    )
    graph_scratch_page = cache.reserve_scratch_page() if graph_decode else None
    launcher = DistTPLauncher(
        model_path,
        tensor_parallel_size,
        num_pages,
        page_size,
        vocab=grammar_vocab,
        graph_scratch_page=graph_scratch_page,
        graph_max_batch=graph_max_batch,
        graph_max_pages=graph_max_pages,
        graph_warmup_iters=graph_warmup_iters,
    )
    generation = load_generation_defaults(model_path)
    # SpeculativeRunner composes over any ModelRunner, and DistTPModelRunner is
    # one: each scoring pass it issues is broadcast like any other step, so every
    # rank replays the identical draft and the m5 D1 agreement invariant holds
    # unchanged. A rejected draft shortens the overlay's outputs, which StateSync
    # already handles — a shrinking output list re-sends the full snapshot.
    active: object = launcher.runner
    if speculative == "ngram":
        active = SpeculativeRunner(active)
    loop = EngineLoop(
        resolved,
        scheduler,
        active,
        default_eos_token_id=generation.eos_token_id,
        default_stop_token_ids=generation.stop_token_ids,
        pipeline_depth=pipeline_depth,
    )
    loop.tp_launcher = launcher  # serve teardown must call launcher.shutdown()
    return loop, cache, scheduler


class KairyuBackend:
    def __init__(
        self,
        num_pages: int = 4096,
        page_size: int = 16,
        max_num_batched_tokens: int = 2048,
        max_num_seqs: int = 256,
        priority_age_s: float | None = 60.0,
        runner: object | None = None,
        tensor_parallel_size: int = 1,
        tokenizer: str | Tokenizer | None = None,
        speculative: str | None = None,
        speculative_tokens: int = 4,
        model_path: str | None = None,
        pd_separation: bool = False,
        pipeline_depth: int = 1,
        decode_mode: str = "eager",
        cuda_graph_max_batch: int = 8,
        cuda_graph_max_pages: int = 512,
        cuda_graph_warmup_iters: int = 3,
    ) -> None:
        self.tensor_parallel_size = tensor_parallel_size
        self._loop, self._cache, self._scheduler = build_engine_loop(
            num_pages=num_pages,
            page_size=page_size,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            priority_age_s=priority_age_s,
            runner=runner,
            tensor_parallel_size=tensor_parallel_size,
            tokenizer=tokenizer,
            speculative=speculative,
            speculative_tokens=speculative_tokens,
            model_path=model_path,
            pd_separation=pd_separation,
            pipeline_depth=pipeline_depth,
            decode_mode=decode_mode,
            cuda_graph_max_batch=cuda_graph_max_batch,
            cuda_graph_max_pages=cuda_graph_max_pages,
            cuda_graph_warmup_iters=cuda_graph_warmup_iters,
        )
        self._queues: dict[str, asyncio.Queue] = {}  # event-loop thread only
        self._active_request_ids: set[str] = set()  # full public-call lifetime
        self._pump_task: asyncio.Task | None = None
        self._engine_error: Exception | None = None  # last step failure, for readiness()

    def validate_request(self, request: GenerationRequest) -> None:
        params = request.sampling_params
        unsupported: list[str] = []
        if params.best_of is not None:
            unsupported.append("best_of")
        if params.prompt_logprobs is not None:
            unsupported.append("prompt_logprobs")
        if not params.skip_special_tokens:
            unsupported.append("skip_special_tokens")
        if not isinstance(params.extra_args, Mapping):
            unsupported.append("extra_args")
        else:
            unsupported.extend(
                f"extra_args.{key}" for key in params.extra_args if key != "response_format"
            )
        for index, tool in enumerate(request.tools):
            function = tool.get("function")
            if isinstance(function, Mapping) and function.get("strict") is True:
                unsupported.append(f"tools[{index}].function.strict")
        if unsupported:
            raise ValueError(
                "Kairyu backend does not support request fields: " + ", ".join(sorted(unsupported))
            )
        self._loop.tokenize_prompt(request.prompt)

    def scheduler_priority_metrics(self) -> dict[str, dict]:
        """Expose bounded native scheduler counters to the serve collector."""

        snapshot = getattr(self._scheduler, "priority_metrics_snapshot", None)
        if snapshot is None:
            return {}
        return snapshot()

    async def _pump(self) -> None:
        restart_after_exit = False
        try:
            while self._loop.has_work():
                updates = await asyncio.to_thread(self._loop.step)
                # a completed step is the only proof the engine still runs, so it
                # is also what clears a previous failure (transient faults must
                # not strand the node out of rotation forever)
                self._engine_error = None
                for request_id, update in updates:
                    queue = self._queues.get(request_id)
                    if queue is not None:
                        queue.put_nowait(update)
        except Exception as error:
            logger.exception("engine step failed")
            # recorded FIRST: purge itself runs through the same broken transport
            # that just failed, and when it raises too the original error escapes
            # as an unretrieved task exception and nothing observes the fault
            self._engine_error = error
            request_ids = tuple(self._queues)
            try:
                await asyncio.to_thread(self._loop.purge, request_ids)
            except Exception:  # noqa: BLE001 - already failing; report the first cause
                pass
            failure = StreamUpdate((), "", True, None, error)
            for request_id in request_ids:
                queue = self._queues.get(request_id)
                if queue is not None:
                    queue.put_nowait(failure)
            # A generic runner exception can be request-local and recover on the
            # next step. A TP step failure is different: one missed collective
            # permanently invalidates the group, so a hot restart loop only floods
            # logs and burns a CPU while every request receives the same 502.
            restart_after_exit = self._loop.has_work() and self.readiness().ready
        finally:
            self._pump_task = None
            if restart_after_exit:
                self._ensure_pump()

    def readiness(self) -> EngineReadiness:
        """Cheap liveness for `/readyz`: KNOWN-FATAL state only, no probe.

        A dead TP rank or a failed TP step is fatal. The group cannot complete a
        trustworthy next collective after either condition and nothing
        in-process can repair its sequence, so the node needs replacing —
        `fatal` says so, and `/health` turns that into a restart signal.

        A step exception deliberately does NOT flip readiness. It is reported for
        diagnosis but cannot be told apart from one bad request, and marking the
        node unready for it is a trap: the load balancer stops sending work, so
        the successful step that would clear the flag never arrives and the node
        stays out of rotation until someone notices (review [P1] on #126).
        """
        launcher = getattr(self._loop, "tp_launcher", None)
        if launcher is not None:
            failure_type = getattr(launcher, "failure_type", lambda: None)()
            if failure_type is not None:
                return EngineReadiness(
                    False,
                    f"tensor-parallel transport failed: {failure_type}",
                    fatal=True,
                )
            dead = launcher.dead_ranks()
            if dead:
                return EngineReadiness(
                    False,
                    f"tensor-parallel ranks not running: {sorted(dead)}",
                    fatal=True,
                )
        return EngineReadiness(True, self._last_error_detail())

    def _last_error_detail(self) -> str:
        """Class name only — this reaches an unauthenticated endpoint, and an
        exception's message can carry an upstream URL or a path (review [P2])."""
        if self._engine_error is None:
            return ""
        return f"last step error: {type(self._engine_error).__name__}"

    def _ensure_pump(self) -> None:
        if not self._loop.has_work():
            return
        if self._pump_task is None or self._pump_task.done():
            self._pump_task = asyncio.get_running_loop().create_task(self._pump())

    def _abort(self, *request_ids: str) -> None:
        for request_id in request_ids:
            self._loop.abort(request_id)
        if not request_ids:
            return
        task = self._pump_task
        if task is None or task.done():
            self._ensure_pump()
        else:
            # The task may have evaluated has_work() just before the abort was
            # enqueued. Re-check after it exits so the op cannot be stranded.
            task.add_done_callback(lambda _task: self._ensure_pump())

    def _reserve_request_ids(self, request_ids: tuple[str, ...]) -> None:
        reserved: set[str] = set()
        for request_id in request_ids:
            if (
                request_id in reserved
                or request_id in self._active_request_ids
                or request_id in self._queues
            ):
                raise ValueError(f"duplicate request_id {request_id!r}")
            reserved.add(request_id)
        self._active_request_ids.update(reserved)

    def _release_request_ids(self, request_ids: tuple[str, ...]) -> None:
        self._active_request_ids.difference_update(request_ids)

    def _remove_queue(self, request_id: str, queue: asyncio.Queue) -> None:
        if self._queues.get(request_id) is queue:
            del self._queues[request_id]

    def _submit(self, request: GenerationRequest, *, pre_reserved: bool = False) -> asyncio.Queue:
        request_id = request.request_id
        if not pre_reserved:
            self._reserve_request_ids((request_id,))
        submitted = False
        queue: asyncio.Queue | None = None
        try:
            self._loop.submit(
                request_id,
                prompt_with_tool_intent(request),
                request.sampling_params,
                priority=request.priority,
                scheduling_class=request.scheduling_class,
            )
            submitted = True
            queue = asyncio.Queue()
            self._queues[request_id] = queue
            if self._pump_task is None:
                self._pump_task = asyncio.get_running_loop().create_task(self._pump())
            return queue
        except BaseException:
            try:
                if submitted:
                    self._abort(request_id)
            finally:
                if queue is not None:
                    self._remove_queue(request_id, queue)
                if not pre_reserved:
                    self._release_request_ids((request_id,))
            raise

    def _result(self, request: GenerationRequest, update: StreamUpdate) -> GenerationResult:
        completion = CompletionOutput(
            index=0,
            text=update.text,
            token_ids=update.outputs,
            cumulative_logprob=update.cumulative_logprob,
            logprobs=update.logprobs,
            finish_reason=update.finish_reason,
            logprob_content=update.logprob_content,
        )
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(completion,),
            finished=update.finished,
            usage=GenerationUsage(
                prompt_tokens=update.num_prompt_tokens,
                completion_tokens=len(update.outputs),
                cached_tokens=update.num_cached_tokens,
            ),
        )

    def _sub_requests(self, request: GenerationRequest) -> list[GenerationRequest]:
        """n>1 as n engine sub-requests (m9 D3). Completion 0 uses the user
        seed IDENTICALLY (reproducibility parity with n=1); i>0 derive via
        splitmix. Siblings prefill independently (documented: same-schedule
        admissions do not share radix pages; n x prompt page pressure)."""
        params = request.sampling_params
        subs = []
        for index in range(params.n):
            seed = params.seed
            if seed is not None and index > 0:
                seed = mix_seed(seed, index)
            subs.append(
                GenerationRequest(
                    request_id=f"{request.request_id}#c{index}",
                    prompt=request.prompt,
                    sampling_params=params.clone(n=1, seed=seed),
                    priority=request.priority,
                    scheduling_class=request.scheduling_class,
                    cache_hint=request.cache_hint,
                    tools=request.tools,
                    tool_choice=request.tool_choice,
                    tools_in_prompt=request.tools_in_prompt,
                )
            )
        return subs

    def _merged(
        self,
        request: GenerationRequest,
        latest: dict[int, StreamUpdate],
        finished: bool,
    ) -> GenerationResult:
        completions = tuple(
            CompletionOutput(
                index=index,
                text=update.text,
                token_ids=update.outputs,
                cumulative_logprob=update.cumulative_logprob,
                logprobs=update.logprobs,
                finish_reason=update.finish_reason,
                logprob_content=update.logprob_content,
            )
            for index, update in sorted(latest.items())
        )
        first = latest.get(0)
        usage = None
        if first is not None:
            # prompt counted ONCE (m9 D1 aggregation rule); completions summed
            usage = GenerationUsage(
                prompt_tokens=first.num_prompt_tokens,
                completion_tokens=sum(len(u.outputs) for u in latest.values()),
                cached_tokens=first.num_cached_tokens,
            )
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=completions,
            finished=finished,
            usage=usage,
        )

    async def _generate_one(
        self, request: GenerationRequest, *, pre_reserved: bool = False
    ) -> GenerationResult:
        queue = self._submit(request, pre_reserved=pre_reserved)
        finished_cleanly = False
        pump_failed = False
        try:
            while True:
                update: StreamUpdate = await queue.get()
                if update.error is not None:
                    pump_failed = True
                    raise update.error
                if update.finished:
                    finished_cleanly = True
                    return self._result(request, update)
        finally:
            if not finished_cleanly and not pump_failed:
                self._abort(request.request_id)
            self._remove_queue(request.request_id, queue)
            if not pre_reserved:
                self._release_request_ids((request.request_id,))

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        if request.sampling_params.n <= 1:
            return await self._generate_one(request)
        subs = self._sub_requests(request)
        request_ids = tuple(sub.request_id for sub in subs)
        self._reserve_request_ids(request_ids)
        tasks: list[asyncio.Task] = []
        try:
            tasks = [
                asyncio.create_task(self._generate_one(sub, pre_reserved=True)) for sub in subs
            ]
            results = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            self._release_request_ids(request_ids)
        latest = {
            index: StreamUpdate(
                outputs=result.completions[0].token_ids,
                text=result.completions[0].text,
                finished=True,
                finish_reason=result.completions[0].finish_reason,
                logprobs=result.completions[0].logprobs,
                cumulative_logprob=result.completions[0].cumulative_logprob or 0.0,
                num_prompt_tokens=result.usage.prompt_tokens if result.usage else 0,
                num_cached_tokens=result.usage.cached_tokens if result.usage else 0,
                logprob_content=result.completions[0].logprob_content,
            )
            for index, result in enumerate(results)
        }
        return self._merged(request, latest, finished=True)

    async def _stream_one(self, request: GenerationRequest) -> AsyncIterator[GenerationResult]:
        queue = self._submit(request)
        emitted = -1
        finished_cleanly = False
        pump_failed = False
        try:
            while True:
                update: StreamUpdate = await queue.get()
                if update.error is not None:
                    pump_failed = True
                    raise update.error
                if update.finished:
                    finished_cleanly = True
                if len(update.outputs) > emitted or update.finished:
                    emitted = len(update.outputs)
                    yield self._result(request, update)
                if update.finished:
                    return
        finally:
            if not finished_cleanly and not pump_failed:
                self._abort(request.request_id)
            self._remove_queue(request.request_id, queue)
            self._release_request_ids((request.request_id,))

    async def stream(self, request: GenerationRequest) -> AsyncIterator[GenerationResult]:
        if request.sampling_params.n <= 1:
            async for result in self._stream_one(request):
                yield result
            return
        # merged n>1 stream: every partial is the cumulative snapshot of ALL
        # completions seen so far (MockBackend semantics — the SSE layer emits
        # finish chunks from the LAST partial's completions)
        subs = self._sub_requests(request)
        request_ids = tuple(sub.request_id for sub in subs)
        self._reserve_request_ids(request_ids)
        queues: dict[int, asyncio.Queue] = {}
        pending: dict[int, asyncio.Future] = {}
        latest: dict[int, StreamUpdate] = {}
        finished: set[int] = set()
        pump_failed = False
        try:
            for index, sub in enumerate(subs):
                queues[index] = self._submit(sub, pre_reserved=True)
            pending = {index: asyncio.ensure_future(queue.get()) for index, queue in queues.items()}
            while len(finished) < len(subs):
                done, _ = await asyncio.wait(pending.values(), return_when=asyncio.FIRST_COMPLETED)
                for index in list(pending):
                    task = pending[index]
                    if task not in done:
                        continue
                    update: StreamUpdate = task.result()
                    if update.error is not None:
                        pump_failed = True
                        raise update.error
                    latest[index] = update
                    if update.finished:
                        finished.add(index)
                        del pending[index]
                    else:
                        pending[index] = asyncio.ensure_future(queues[index].get())
                yield self._merged(request, latest, finished=len(finished) == len(subs))
        except BaseException:
            if not pump_failed:
                self._abort(*(sub.request_id for sub in subs))
            raise
        finally:
            for task in pending.values():
                task.cancel()
            for index, queue in queues.items():
                self._remove_queue(subs[index].request_id, queue)
            self._release_request_ids(request_ids)

    async def shutdown(self) -> None:
        if self._pump_task is not None:
            self._pump_task.cancel()
        await asyncio.to_thread(self._loop.close)
        # stop the spawned TP worker ranks (no-op unless --tp N launched them)
        launcher = getattr(self._loop, "tp_launcher", None)
        if launcher is not None:
            await asyncio.to_thread(launcher.shutdown)


register_backend("kairyu", KairyuBackend)
