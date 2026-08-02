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
import threading
import weakref
from collections.abc import AsyncIterator, Mapping
from pathlib import Path

from kairyu.engine.backend import (
    EngineReadiness,
    GenerationRequest,
    GenerationResult,
    GenerationUsage,
    prompt_with_tool_intent,
    validate_native_request_surface,
)
from kairyu.engine.core.comm import FakeCommunicator
from kairyu.engine.core.kv_cache_dtype import (
    kv_cache_dtype_name,
    resolve_kv_cache_dtype,
    validate_kv_cache_dtype,
)
from kairyu.engine.core.pd_loop import PDLoopAdapter
from kairyu.engine.core.radix_kv import RadixKVCache
from kairyu.engine.core.sampling_types import SampledToken, mix_seed
from kairyu.engine.core.scheduler import ScheduledChunk, Scheduler
from kairyu.engine.core.spec_runner import SpeculativeRunner
from kairyu.engine.core.tp_runner import TPModelRunner, validate_tp_degree
from kairyu.engine.engine_loop import (
    EngineLoop,
    PreparedPrompt,
    StreamUpdate,
    _validate_max_model_len,
)
from kairyu.engine.prompt import supplied_prompt_token_ids
from kairyu.engine.registry import register_backend
from kairyu.engine.tokenizer import (
    Tokenizer,
    grammar_vocabulary,
    resolve_tokenizer,
)
from kairyu.outputs import CompletionOutput

_VOCAB_SIZE = 50_000
_DECODE_MODES = frozenset({"eager", "cuda_graph"})
_EXPERT_PARALLEL_SIZES = frozenset({1, 2, 4})
logger = logging.getLogger(__name__)


def _graph_row_capacity(
    request_batch_capacity: int,
    token_budget: int,
    *,
    speculative: bool,
    speculative_tokens: int,
) -> int:
    """Translate request capacity to the flattened decode-row capacity."""
    if not speculative:
        return request_batch_capacity
    return min(
        token_budget,
        request_batch_capacity * (speculative_tokens + 1),
    )


class _ToyRunner:
    """Deterministic CPU stand-in for the GPU model forward (greedy only —
    sampling params take effect with a Sampler-equipped runner, m8 D2)."""

    supports_batched_verification = True

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
                count = 1 if chunk.is_prefill else chunk.num_tokens
                sampled[chunk.request_id] = tuple(
                    SampledToken(
                        (seed + 31 * (chunk.position + offset)) % _VOCAB_SIZE
                    )
                    for offset in range(count)
                )
        return sampled

    def execute_passive(
        self, scheduled: tuple[ScheduledChunk, ...], states: Mapping[str, object]
    ) -> dict[str, tuple[SampledToken, ...]]:
        """Toy has no model/KV work, but followers still enter the passive seam."""
        return {}

    @staticmethod
    def _chunk_emits_token(chunk: ScheduledChunk, state: object) -> bool:
        return not chunk.is_prefill or state.prefill_done

    @staticmethod
    def _chunk_packet_width(chunk: ScheduledChunk) -> int:
        """Fixed packet slots derivable from the already-broadcast chunk."""
        return 1 if chunk.is_prefill else chunk.num_tokens

    def make_sampling_token_packet(
        self,
        scheduled: tuple[ScheduledChunk, ...],
        states: Mapping[str, object],
        sampled: Mapping[str, tuple[SampledToken, ...]] | None = None,
    ):
        """Build the same fixed variable-width layout as ``PagedModelRunner``."""
        import torch

        chunks = tuple(scheduled)
        packet = torch.full(
            (sum(self._chunk_packet_width(chunk) for chunk in chunks),),
            -1,
            dtype=torch.int64,
        )
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
        offset = 0
        for chunk in chunks:
            width = self._chunk_packet_width(chunk)
            if chunk.request_id not in expected:
                offset += width
                continue
            records = sampled[chunk.request_id]
            if len(records) != width or any(record.token_id < 0 for record in records):
                raise RuntimeError(
                    "TP sampling owner emitted an invalid token sequence for "
                    f"{chunk.request_id!r} at position {chunk.position}: "
                    f"expected {width}, got {len(records)}"
                )
            packet[offset : offset + width] = torch.tensor(
                [record.token_id for record in records], dtype=torch.int64
            )
            offset += width
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
        expected_length = sum(self._chunk_packet_width(chunk) for chunk in chunks)
        if packet.numel() != expected_length:
            raise RuntimeError(
                "TP sampling packet length does not match the scheduled layout: "
                f"got {packet.numel()}, expected {expected_length}"
            )
        offset = 0
        for chunk in chunks:
            width = self._chunk_packet_width(chunk)
            emits = self._chunk_emits_token(chunk, states[chunk.request_id])
            values = tuple(int(value) for value in packet[offset : offset + width])
            if emits and any(token_id < 0 for token_id in values):
                raise RuntimeError(
                    "TP sampling packet is missing an authoritative token for "
                    f"{chunk.request_id!r} at position {chunk.position}"
                )
            if not emits and any(token_id != -1 for token_id in values):
                raise RuntimeError(
                    "TP sampling packet emitted an unexpected token for partial "
                    f"prefill {chunk.request_id!r} at position {chunk.position}"
                )
            if emits:
                future = self._future_tokens.setdefault(chunk.request_id, {})
                for index, token_id in enumerate(values):
                    future[chunk.position + index] = token_id
            offset += width

    def release(self, request_id: str) -> None:
        self._future_tokens.pop(request_id, None)


def build_engine_loop(
    *,
    num_pages: int = 4096,
    page_size: int = 16,
    max_num_batched_tokens: int = 2048,
    max_num_seqs: int = 256,
    max_model_len: int | None = None,
    priority_age_s: float | None = 60.0,
    runner: object | None = None,
    tensor_parallel_size: int = 1,
    expert_parallel_size: int = 1,
    expert_parallel_attention_dp: bool = False,
    tokenizer: str | Tokenizer | None = None,
    speculative: str | None = None,
    speculative_tokens: int = 4,
    model_path: str | None = None,
    pd_separation: bool = False,
    pd_prefill_device: str | None = None,
    pd_decode_device: str | None = None,
    pd_defer_handoff: bool = True,
    pipeline_depth: int = 1,
    decode_mode: str = "eager",
    cuda_graph_max_batch: int = 8,
    cuda_graph_max_pages: int = 512,
    cuda_graph_warmup_iters: int = 3,
    kv_cache_dtype: str = "auto",
    dram_kv_tier_capacity_pages: int = 0,
    dram_kv_tier_profile: str | Path | None = None,
) -> tuple[EngineLoop, RadixKVCache, Scheduler | PDLoopAdapter]:
    """Assemble the engine stack; shared by KairyuBackend and the ZMQ service.

    ``model_path`` loads a real checkpoint (m12 D5): DenseDecoder +
    PagedKVPool + PagedModelRunner + Sampler, tokenizer from the same dir
    unless overridden. Mutually exclusive with ``runner``. Real-model TP > 1
    and EP > 1 spawn their respective distributed launcher; the loop's
    ``.parallel_launcher`` handle must be ``shutdown()`` on serve teardown. A
    caller-supplied ``runner`` is a single rank-local object and therefore
    requires both distributed sizes to remain 1.
    """
    _validate_max_model_len(max_model_len)
    if type(tensor_parallel_size) is not int or tensor_parallel_size < 1:
        raise ValueError(
            "tensor_parallel_size must be a positive integer; "
            f"got {tensor_parallel_size!r}"
        )
    if (
        type(expert_parallel_size) is not int
        or expert_parallel_size not in _EXPERT_PARALLEL_SIZES
    ):
        supported = ", ".join(str(size) for size in sorted(_EXPERT_PARALLEL_SIZES))
        raise ValueError(
            f"expert_parallel_size must be one of {supported}; "
            f"got {expert_parallel_size!r}"
        )
    expert_parallel = expert_parallel_size > 1
    if type(expert_parallel_attention_dp) is not bool:
        raise TypeError("expert_parallel_attention_dp must be a boolean")
    if expert_parallel_attention_dp and expert_parallel_size != 4:
        raise ValueError(
            "expert_parallel_attention_dp requires expert_parallel_size=4"
        )
    if expert_parallel and tensor_parallel_size > 1:
        raise ValueError(
            "tensor_parallel_size > 1 and expert_parallel_size > 1 are "
            "mutually exclusive"
        )
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
    kv_cache_dtype = validate_kv_cache_dtype(kv_cache_dtype)
    if kv_cache_dtype != "auto" and model_path is None:
        raise ValueError(
            f"kv_cache_dtype={kv_cache_dtype!r} requires a real model_path; "
            "custom and toy runners do not expose a managed KV pool"
        )
    if kv_cache_dtype != "auto" and pd_separation:
        raise ValueError(
            "explicit kv_cache_dtype does not support P-D separation"
        )
    if (
        type(dram_kv_tier_capacity_pages) is not int
        or dram_kv_tier_capacity_pages < 0
    ):
        raise ValueError(
            "dram_kv_tier_capacity_pages must be a non-negative integer"
        )
    if dram_kv_tier_profile is not None and not isinstance(
        dram_kv_tier_profile, (str, Path)
    ):
        raise ValueError("dram_kv_tier_profile must be a local path or null")
    if dram_kv_tier_profile is not None and not str(dram_kv_tier_profile):
        raise ValueError(
            "dram_kv_tier_profile must be a non-empty local path or null"
        )
    if (dram_kv_tier_capacity_pages > 0) != (dram_kv_tier_profile is not None):
        raise ValueError(
            "DRAM KV tier requires both a positive "
            "dram_kv_tier_capacity_pages and dram_kv_tier_profile"
        )
    if dram_kv_tier_capacity_pages:
        if model_path is None:
            raise ValueError("DRAM KV tier requires a real model_path")
        if pd_separation:
            raise ValueError("DRAM KV tier does not support P-D separation")
    if not pd_separation and (
        pd_prefill_device is not None
        or pd_decode_device is not None
        or pd_defer_handoff is not True
    ):
        raise ValueError(
            "pd_prefill_device, pd_decode_device, and a non-default "
            "pd_defer_handoff require pd_separation=True"
        )
    if expert_parallel:
        if model_path is None:
            raise ValueError("expert parallelism requires a real model_path")
        if not expert_parallel_attention_dp and pipeline_depth != 1:
            raise ValueError(
                "replicated-attention expert parallelism requires pipeline_depth=1"
            )
        if decode_mode != "eager" and not expert_parallel_attention_dp:
            raise ValueError(
                "replicated-attention expert parallelism requires "
                "decode_mode='eager'"
            )
        if kv_cache_dtype != "bfloat16":
            raise ValueError(
                "expert parallelism requires kv_cache_dtype='bfloat16'"
            )
        if pd_separation:
            raise ValueError("expert parallelism does not support P-D separation")
        if speculative is not None:
            raise ValueError(
                "expert parallelism does not support speculative decoding"
            )
        if dram_kv_tier_capacity_pages:
            raise ValueError("expert parallelism does not support a DRAM KV tier")
        return _build_dist_ep_loop(
            model_path=model_path,
            expert_parallel_size=expert_parallel_size,
            expert_parallel_attention_dp=expert_parallel_attention_dp,
            num_pages=num_pages,
            page_size=page_size,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            priority_age_s=priority_age_s,
            tokenizer=tokenizer,
            pipeline_depth=pipeline_depth,
            decode_mode=decode_mode,
            cuda_graph_max_batch=cuda_graph_max_batch,
            cuda_graph_max_pages=cuda_graph_max_pages,
            cuda_graph_warmup_iters=cuda_graph_warmup_iters,
            kv_cache_dtype=kv_cache_dtype,
        )
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
            max_model_len=max_model_len,
            priority_age_s=priority_age_s,
            tokenizer=tokenizer,
            pipeline_depth=pipeline_depth,
            prefill_device=pd_prefill_device,
            decode_device=pd_decode_device,
            defer_handoff=pd_defer_handoff,
        )

    if model_path is not None and tensor_parallel_size > 1:
        return _build_dist_tp_loop(
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            num_pages=num_pages,
            page_size=page_size,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            priority_age_s=priority_age_s,
            tokenizer=tokenizer,
            speculative=speculative,
            speculative_tokens=speculative_tokens,
            pipeline_depth=pipeline_depth,
            graph_decode=graph_decode,
            graph_max_batch=cuda_graph_max_batch,
            graph_max_pages=cuda_graph_max_pages,
            graph_warmup_iters=cuda_graph_warmup_iters,
            kv_cache_dtype=kv_cache_dtype,
            dram_kv_tier_capacity_pages=dram_kv_tier_capacity_pages,
            dram_kv_tier_profile=dram_kv_tier_profile,
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
    attention_backend_decision = None
    resolved_kv_cache_dtype = None
    if model_path is not None:
        import torch

        from kairyu.engine.core.attention import select_backend
        from kairyu.engine.core.attention_selector import (
            attention_backend_execution_identity,
        )
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
        attention_backend = select_backend(profile, device=compute_device)
        attention_backend_decision = attention_backend.selection_decision
        model, model_config, generation = load_model(
            model_path,
            dtype=compute_dtype,
            attention_backend=attention_backend,
            target_device=compute_device,
        )
        resolved_kv_cache_dtype = resolve_kv_cache_dtype(
            kv_cache_dtype,
            compute_dtype,
            profile,
            attention_backend,
            model_config,
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
    dram_kv_binding = None
    if model_path is not None:
        pool = PagedKVPool.for_cache(
            cache,
            model_config,
            dtype=resolved_kv_cache_dtype,
            device=compute_device,
        )
        if dram_kv_tier_profile is not None:
            from kairyu.engine.core.kv_tier_policy import build_dram_kv_tier

            dram_kv_binding = build_dram_kv_tier(
                pool,
                model_path=model_path,
                tensor_parallel_size=1,
                tensor_parallel_rank=0,
                capacity_pages=dram_kv_tier_capacity_pages,
                profile_path=dram_kv_tier_profile,
                attention_backend_identity=attention_backend_execution_identity(
                    attention_backend_decision,
                    attention_backend,
                ),
                max_num_batched_tokens=max_num_batched_tokens,
            )
        graph_options = {}
        if graph_decode:
            from kairyu.engine.core.cuda_graph_gpu import CudaGraphBackend

            # ``cuda_graph_max_batch`` is a request-batch capacity.  Target
            # verification flattens each speculative request into at most
            # k+1 decode rows, and those rows must fit a graph bucket too.
            graph_row_capacity = _graph_row_capacity(
                cuda_graph_max_batch,
                max_num_batched_tokens,
                speculative=speculative is not None,
                speculative_tokens=speculative_tokens,
            )
            graph_options = {
                "graph_backend": CudaGraphBackend(warmup_iters=cuda_graph_warmup_iters),
                "graph_max_batch": graph_row_capacity,
                "graph_max_pages": cuda_graph_max_pages,
            }
        runner = PagedModelRunner(
            model,
            pool,
            sampler=Sampler(vocab_provider=lambda: grammar_vocab),
            cache=cache,
            **graph_options,
        )
        if dram_kv_binding is not None:
            runner.dram_kv_tier = dram_kv_binding.tier
            runner.dram_kv_policy = dram_kv_binding.policy
            cache.attach_dram_tier(
                dram_kv_binding.tier,
                min_restore_tokens=dram_kv_binding.policy.min_restore_tokens,
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
        max_model_len=max_model_len,
    )
    loop.parallel_launcher = None  # single-process: nothing to tear down
    loop.tp_launcher = None  # compatibility alias for TP-specific callers
    loop.attention_backend_decision = attention_backend_decision
    loop.kv_cache_dtype_requested = kv_cache_dtype
    loop.kv_cache_dtype_resolved = (
        kv_cache_dtype_name(resolved_kv_cache_dtype)
        if resolved_kv_cache_dtype is not None
        else "not-applicable"
    )
    loop.dram_kv_tier_enabled = dram_kv_binding is not None
    loop.dram_kv_tier_capacity_pages = (
        dram_kv_binding.tier.capacity_pages
        if dram_kv_binding is not None
        else 0
    )
    loop.dram_kv_tier_profile_sha256 = (
        dram_kv_binding.policy.profile_sha256
        if dram_kv_binding is not None
        else None
    )
    loop.dram_kv_tier_min_restore_tokens = (
        dram_kv_binding.policy.min_restore_tokens
        if dram_kv_binding is not None
        else None
    )
    return loop, cache, scheduler


def _build_pd_loop(
    *,
    model_path: str,
    num_pages: int,
    page_size: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    max_model_len: int | None,
    priority_age_s: float | None,
    tokenizer: str | Tokenizer | None,
    pipeline_depth: int,
    prefill_device: str | None,
    decode_device: str | None,
    defer_handoff: bool,
) -> tuple[EngineLoop, RadixKVCache, PDLoopAdapter]:
    """Serve through a prefill/decode pair (m18 D3, G2 stage 5.3).

    `PDCoordinator` had no serving entry point: `pd_factory` builds it, but a
    deployment could only reach it from Python. This is the `backend: kairyu`
    option that turns it on, so `pd_separation: true` in a deployment YAML is a
    real topology rather than a config surface m2 §2.4 reserved and never wired.
    `pd_prefill_device` and `pd_decode_device` select role GPUs;
    `pd_defer_handoff=false` selects the synchronized control. Deferred
    cross-device handoff requires CUDA peer access and publishes only after
    physical event completion.

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
        prefill_device=prefill_device,
        decode_device=decode_device,
        defer_handoff=defer_handoff,
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
        max_model_len=max_model_len,
    )
    loop.parallel_launcher = None
    loop.tp_launcher = None
    loop.pd_coordinator = coordinator
    loop.attention_backend_decision = getattr(
        coordinator, "attention_backend_decision", None
    )
    # P-D currently accepts only the default cache policy; each role owns and
    # reports its own resolved pool dtype.
    loop.kv_cache_dtype_requested = "auto"
    loop.kv_cache_dtype_resolved = "role-specific"
    loop.dram_kv_tier_enabled = False
    loop.dram_kv_tier_capacity_pages = 0
    loop.dram_kv_tier_profile_sha256 = None
    loop.dram_kv_tier_min_restore_tokens = None
    return loop, adapter.kv_cache, adapter


def _build_dist_ep_loop(
    *,
    model_path: str,
    expert_parallel_size: int,
    expert_parallel_attention_dp: bool = False,
    num_pages: int,
    page_size: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    max_model_len: int | None,
    priority_age_s: float | None,
    tokenizer: str | Tokenizer | None,
    pipeline_depth: int = 1,
    decode_mode: str = "eager",
    cuda_graph_max_batch: int = 8,
    cuda_graph_max_pages: int = 512,
    cuda_graph_warmup_iters: int = 3,
    kv_cache_dtype: str = "bfloat16",
) -> tuple[EngineLoop, RadixKVCache, Scheduler]:
    """Build the production replicated-attention or attention-DP EP loop."""

    from kairyu.engine.core.worker import DistEPLauncher
    from kairyu.models.loader import load_generation_defaults

    resolved = resolve_tokenizer(tokenizer if tokenizer is not None else model_path)
    raw_config = json.loads((Path(model_path) / "config.json").read_text())
    model_vocab_size = int(raw_config["vocab_size"])
    grammar_vocab = grammar_vocabulary(
        resolved,
        model_vocab_size=model_vocab_size,
    )
    generation = load_generation_defaults(model_path)
    cache = RadixKVCache(num_pages=num_pages, page_size=page_size)
    scheduler = Scheduler(
        cache,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        page_size=page_size,
        speculative_tokens=0,
        priority_age_s=priority_age_s,
    )
    graph_decode = decode_mode == "cuda_graph"
    graph_row_capacity = _graph_row_capacity(
        cuda_graph_max_batch,
        max_num_batched_tokens,
        speculative=False,
        speculative_tokens=0,
    )
    # Attention-DP ranks own a physical pool larger than the scheduler-visible
    # namespace: two prefill scratch pages, one unique decode scratch page per
    # maximum graph row, then this distinct graph-capture scratch page.
    graph_scratch_page = (
        num_pages + 2 + graph_row_capacity if graph_decode else None
    )
    launcher = DistEPLauncher(
        model_path,
        expert_parallel_size,
        num_pages,
        page_size,
        vocab=grammar_vocab,
        attention_dp=expert_parallel_attention_dp,
        pipeline_depth=pipeline_depth,
        decode_mode=decode_mode,
        kv_cache_dtype=kv_cache_dtype,
        pd_separation=False,
        graph_scratch_page=graph_scratch_page,
        graph_max_batch=graph_row_capacity if graph_decode else 0,
        graph_max_pages=cuda_graph_max_pages if graph_decode else 0,
        graph_warmup_iters=cuda_graph_warmup_iters,
        dram_kv_tier_capacity_pages=0,
        dram_kv_tier_profile=None,
        speculative=None,
    )
    try:
        loop = EngineLoop(
            resolved,
            scheduler,
            launcher.runner,
            default_eos_token_id=generation.eos_token_id,
            default_stop_token_ids=generation.stop_token_ids,
            pipeline_depth=pipeline_depth,
            max_model_len=max_model_len,
        )
    except BaseException:
        launcher.shutdown()
        raise
    loop.parallel_launcher = launcher
    loop.ep_launcher = launcher
    loop.tp_launcher = None
    loop.parallelism_metadata = dict(launcher.parallelism_metadata())
    loop.attention_backend_decision = launcher.attention_backend_decision
    loop.kv_cache_dtype_requested = launcher.kv_cache_dtype_requested
    loop.kv_cache_dtype_resolved = launcher.kv_cache_dtype_resolved
    loop.dram_kv_tier_enabled = False
    loop.dram_kv_tier_capacity_pages = 0
    loop.dram_kv_tier_profile_sha256 = None
    loop.dram_kv_tier_min_restore_tokens = None
    return loop, cache, scheduler


def _build_dist_tp_loop(
    *,
    model_path: str,
    tensor_parallel_size: int,
    num_pages: int,
    page_size: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    max_model_len: int | None,
    priority_age_s: float | None,
    tokenizer: str | Tokenizer | None,
    speculative: str | None = None,
    speculative_tokens: int = 4,
    pipeline_depth: int = 1,
    graph_decode: bool = False,
    graph_max_batch: int = 8,
    graph_max_pages: int = 512,
    graph_warmup_iters: int = 3,
    kv_cache_dtype: str = "auto",
    dram_kv_tier_capacity_pages: int = 0,
    dram_kv_tier_profile: str | Path | None = None,
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
    graph_row_capacity = _graph_row_capacity(
        graph_max_batch,
        max_num_batched_tokens,
        speculative=graph_decode and speculative is not None,
        speculative_tokens=speculative_tokens,
    )
    launcher = DistTPLauncher(
        model_path,
        tensor_parallel_size,
        num_pages,
        page_size,
        vocab=grammar_vocab,
        graph_scratch_page=graph_scratch_page,
        graph_max_batch=graph_row_capacity,
        graph_max_pages=graph_max_pages,
        graph_warmup_iters=graph_warmup_iters,
        kv_cache_dtype=kv_cache_dtype,
        dram_kv_tier_capacity_pages=dram_kv_tier_capacity_pages,
        dram_kv_tier_profile=dram_kv_tier_profile,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    if launcher.dram_kv_binding is not None:
        cache.attach_dram_tier(
            launcher.runner,
            min_restore_tokens=(
                launcher.dram_kv_binding.policy.min_restore_tokens
            ),
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
        max_model_len=max_model_len,
    )
    loop.parallel_launcher = launcher
    loop.tp_launcher = launcher  # compatibility alias for TP-specific callers
    loop.attention_backend_decision = launcher.attention_backend_decision
    loop.kv_cache_dtype_requested = launcher.kv_cache_dtype_requested
    loop.kv_cache_dtype_resolved = launcher.kv_cache_dtype_resolved
    loop.dram_kv_tier_enabled = launcher.dram_kv_binding is not None
    loop.dram_kv_tier_capacity_pages = (
        launcher.dram_kv_binding.tier.capacity_pages
        if launcher.dram_kv_binding is not None
        else 0
    )
    loop.dram_kv_tier_profile_sha256 = (
        launcher.dram_kv_binding.policy.profile_sha256
        if launcher.dram_kv_binding is not None
        else None
    )
    loop.dram_kv_tier_min_restore_tokens = (
        launcher.dram_kv_binding.policy.min_restore_tokens
        if launcher.dram_kv_binding is not None
        else None
    )
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
        pd_prefill_device: str | None = None,
        pd_decode_device: str | None = None,
        pd_defer_handoff: bool = True,
        max_model_len: int | None = None,
        kv_cache_dtype: str = "auto",
        dram_kv_tier_capacity_pages: int = 0,
        dram_kv_tier_profile: str | Path | None = None,
        expert_parallel_size: int = 1,
        expert_parallel_attention_dp: bool = False,
    ) -> None:
        self.tensor_parallel_size = tensor_parallel_size
        self.expert_parallel_size = expert_parallel_size
        self.expert_parallel_attention_dp = expert_parallel_attention_dp
        self._loop, self._cache, self._scheduler = build_engine_loop(
            num_pages=num_pages,
            page_size=page_size,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            priority_age_s=priority_age_s,
            runner=runner,
            tensor_parallel_size=tensor_parallel_size,
            expert_parallel_size=expert_parallel_size,
            expert_parallel_attention_dp=expert_parallel_attention_dp,
            tokenizer=tokenizer,
            speculative=speculative,
            speculative_tokens=speculative_tokens,
            model_path=model_path,
            pd_separation=pd_separation,
            pd_prefill_device=pd_prefill_device,
            pd_decode_device=pd_decode_device,
            pd_defer_handoff=pd_defer_handoff,
            pipeline_depth=pipeline_depth,
            decode_mode=decode_mode,
            cuda_graph_max_batch=cuda_graph_max_batch,
            cuda_graph_max_pages=cuda_graph_max_pages,
            cuda_graph_warmup_iters=cuda_graph_warmup_iters,
            kv_cache_dtype=kv_cache_dtype,
            dram_kv_tier_capacity_pages=dram_kv_tier_capacity_pages,
            dram_kv_tier_profile=dram_kv_tier_profile,
        )
        self.attention_backend_decision = getattr(
            self._loop, "attention_backend_decision", None
        )
        loop_parallelism_metadata = getattr(
            self._loop,
            "parallelism_metadata",
            None,
        )
        self.parallelism_metadata = (
            dict(loop_parallelism_metadata)
            if isinstance(loop_parallelism_metadata, Mapping)
            else None
        )
        self.kv_cache_dtype_requested = self._loop.kv_cache_dtype_requested
        self.kv_cache_dtype_resolved = self._loop.kv_cache_dtype_resolved
        self.dram_kv_tier_enabled = self._loop.dram_kv_tier_enabled
        self.dram_kv_tier_capacity_pages = (
            self._loop.dram_kv_tier_capacity_pages
        )
        self.dram_kv_tier_profile_sha256 = (
            self._loop.dram_kv_tier_profile_sha256
        )
        self.dram_kv_tier_min_restore_tokens = (
            self._loop.dram_kv_tier_min_restore_tokens
        )
        self._queues: dict[str, asyncio.Queue] = {}  # event-loop thread only
        self._active_request_ids: set[str] = set()  # full public-call lifetime
        self._pump_task: asyncio.Task | None = None
        self._engine_error: Exception | None = None  # last step failure, for readiness()
        self._shutdown_started = False
        self._shutdown_task: asyncio.Task[None] | None = None
        # A native HTTP preflight resolves text to enforce max_model_len before
        # response headers. Keep that result only for this exact request object
        # and consume it on submit; this is not a cross-request prompt cache.
        self._prepared_requests: dict[
            int,
            tuple[weakref.ReferenceType[GenerationRequest], PreparedPrompt],
        ] = {}
        self._prepared_requests_lock = threading.Lock()

    def parallelism_metadata_snapshot(self) -> dict[str, object] | None:
        """Refresh topology counters for diagnostics without a rank collective."""

        launcher = getattr(self._loop, "parallel_launcher", None)
        getter = getattr(launcher, "parallelism_metadata", None)
        if callable(getter):
            metadata = getter()
            return dict(metadata) if isinstance(metadata, Mapping) else None
        metadata = self.parallelism_metadata
        return dict(metadata) if isinstance(metadata, Mapping) else None

    def _peek_prepared_request(
        self,
        request: GenerationRequest,
    ) -> PreparedPrompt | None:
        key = id(request)
        with self._prepared_requests_lock:
            cached = self._prepared_requests.get(key)
            if cached is None:
                return None
            if cached[0]() is request:
                return cached[1]
            # Defensive against delayed weakref callbacks and object-ID reuse.
            self._prepared_requests.pop(key, None)
        return None

    def _retain_prepared_request(
        self,
        request: GenerationRequest,
        prepared: PreparedPrompt,
    ) -> PreparedPrompt:
        key = id(request)
        backend_ref = weakref.ref(self)

        def discard(request_ref: weakref.ReferenceType[GenerationRequest]) -> None:
            backend = backend_ref()
            if backend is None:
                return
            with backend._prepared_requests_lock:
                current = backend._prepared_requests.get(key)
                if current is not None and current[0] is request_ref:
                    backend._prepared_requests.pop(key, None)

        request_ref = weakref.ref(request, discard)
        with self._prepared_requests_lock:
            cached = self._prepared_requests.get(key)
            if cached is not None and cached[0]() is request:
                return cached[1]
            self._prepared_requests[key] = (request_ref, prepared)
        return prepared

    def _take_prepared_request(
        self,
        request: GenerationRequest,
    ) -> PreparedPrompt | None:
        key = id(request)
        with self._prepared_requests_lock:
            cached = self._prepared_requests.get(key)
            if cached is None:
                return None
            if cached[0]() is request:
                self._prepared_requests.pop(key, None)
                return cached[1]
            self._prepared_requests.pop(key, None)
        return None

    def _prepare_request(self, request: GenerationRequest) -> PreparedPrompt:
        return self._loop.prepare_prompt(
            prompt_with_tool_intent(request),
            request.sampling_params,
        )

    def _take_or_prepare_request(
        self,
        request: GenerationRequest,
    ) -> PreparedPrompt:
        prepared = self._take_prepared_request(request)
        if prepared is not None:
            return prepared
        return self._prepare_request(request)

    def validate_request(self, request: GenerationRequest) -> None:
        validate_native_request_surface(request)
        if self._peek_prepared_request(request) is None:
            prepared = self._prepare_request(request)
            self._retain_prepared_request(request, prepared)

    def scheduler_priority_metrics(self) -> dict[str, dict]:
        """Expose bounded native scheduler counters to the serve collector."""

        snapshot = getattr(self._scheduler, "priority_metrics_snapshot", None)
        if snapshot is None:
            return {}
        return snapshot()

    def _publish_step(
        self,
        updates: list[tuple[str, StreamUpdate]],
    ) -> None:
        """Deliver one completed step on the owning asyncio event loop."""

        # A completed step is the only proof the engine still runs, so it also
        # clears a previous transient failure without cross-thread state writes.
        self._engine_error = None
        for request_id, update in updates:
            queue = self._queues.get(request_id)
            if queue is not None:
                queue.put_nowait(update)

    @staticmethod
    def _mark_delivery_drained(delivery_drained: asyncio.Future) -> None:
        if not delivery_drained.done():
            delivery_drained.set_result(None)

    def _step_worker(
        self,
        event_loop: asyncio.AbstractEventLoop,
        stop: threading.Event,
        delivery_drained: asyncio.Future,
    ) -> Exception | None:
        """Own ``EngineLoop.step`` until idle or stopped at a step boundary."""

        error: Exception | None = None
        try:
            while not stop.is_set():
                if not self._loop.has_work() or stop.is_set():
                    break
                updates = self._loop.step()
                event_loop.call_soon_threadsafe(self._publish_step, updates)
        except Exception as caught:  # noqa: BLE001 - returned to the event loop
            error = caught
        finally:
            # This marker is queued after every step publication from this
            # worker. Awaiting it prevents a later failure from overtaking a
            # previously completed step's stream update.
            event_loop.call_soon_threadsafe(
                self._mark_delivery_drained,
                delivery_drained,
            )
        return error

    @staticmethod
    async def _settle_worker(worker: asyncio.Task) -> Exception | None:
        """Wait through repeated pump cancellation until the step boundary."""

        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
        return worker.result()

    @staticmethod
    async def _settle_delivery(delivery_drained: asyncio.Future) -> None:
        while not delivery_drained.done():
            try:
                await asyncio.shield(delivery_drained)
            except asyncio.CancelledError:
                continue

    async def _report_step_failure(self, error: Exception) -> None:
        logger.exception(
            "engine step failed",
            exc_info=(type(error), error, error.__traceback__),
        )
        # Record FIRST: purge itself runs through the same broken transport that
        # just failed, and when it raises too the original error must still reach
        # every public request.
        self._engine_error = error
        request_ids = tuple(self._queues)
        cancelled: asyncio.CancelledError | None = None
        purge = asyncio.create_task(asyncio.to_thread(self._loop.purge, request_ids))
        try:
            await asyncio.shield(purge)
        except asyncio.CancelledError as caught:
            cancelled = caught
            while not purge.done():
                try:
                    await asyncio.shield(purge)
                except asyncio.CancelledError:
                    continue
            try:
                purge.result()
            except Exception:  # noqa: BLE001 - preserve the step failure
                pass
        except Exception:  # noqa: BLE001 - already failing; report the first cause
            pass
        failure = StreamUpdate((), "", True, None, error)
        for request_id in request_ids:
            queue = self._queues.get(request_id)
            if queue is not None:
                queue.put_nowait(failure)
        if cancelled is not None:
            raise cancelled

    async def _pump(self) -> None:
        event_loop = asyncio.get_running_loop()
        stop = threading.Event()
        delivery_drained = event_loop.create_future()
        worker = event_loop.create_task(
            asyncio.to_thread(
                self._step_worker,
                event_loop,
                stop,
                delivery_drained,
            )
        )
        cancelled: asyncio.CancelledError | None = None
        worker_error: Exception | None = None
        failure_reported = False
        try:
            try:
                worker_error = await asyncio.shield(worker)
                await asyncio.shield(delivery_drained)
            except asyncio.CancelledError as error:
                cancelled = error
                stop.set()
                worker_error = await self._settle_worker(worker)
                await self._settle_delivery(delivery_drained)
            if worker_error is not None and not self._shutdown_started:
                failure_reported = True
                await self._report_step_failure(worker_error)
        except asyncio.CancelledError as error:
            if cancelled is None:
                cancelled = error
            stop.set()
        finally:
            stop.set()
            if not worker.done():
                worker_error = await self._settle_worker(worker)
            await self._settle_delivery(delivery_drained)
            if (
                worker_error is not None
                and not failure_reported
                and not self._shutdown_started
            ):
                await self._report_step_failure(worker_error)
            current = asyncio.current_task()
            if self._pump_task is current:
                self._pump_task = None
            # A producer can append an op after the worker's final has_work()
            # snapshot. Re-check on the event loop after clearing the old task.
            # Explicit cancellation waits for a later submit/abort to restart;
            # fatal distributed state and shutdown never hot-loop.
            if (
                cancelled is None
                and not self._shutdown_started
                and self._loop.has_work()
                and self.readiness().ready
            ):
                self._ensure_pump()
        if cancelled is not None:
            raise cancelled

    def readiness(self) -> EngineReadiness:
        """Cheap liveness for `/readyz`: KNOWN-FATAL state only, no probe.

        A dead distributed rank or failed transport is fatal. The group cannot
        complete a trustworthy next collective after either condition and
        nothing in-process can repair its sequence, so the node needs replacing
        — `fatal` says so, and `/health` turns that into a restart signal.

        A step exception deliberately does NOT flip readiness. It is reported for
        diagnosis but cannot be told apart from one bad request, and marking the
        node unready for it is a trap: the load balancer stops sending work, so
        the successful step that would clear the flag never arrives and the node
        stays out of rotation until someone notices (review [P1] on #126).
        """
        launcher = getattr(self._loop, "parallel_launcher", None)
        if launcher is None:
            # Compatibility for loops/tests created before the topology-neutral
            # launcher handle was introduced.
            launcher = getattr(self._loop, "tp_launcher", None)
        if launcher is not None:
            parallelism = getattr(self._loop, "parallelism_metadata", None)
            topology = (
                "expert-parallel"
                if isinstance(parallelism, Mapping)
                and parallelism.get("parallelism") == "expert_parallel"
                else "tensor-parallel"
            )
            failure_type = getattr(launcher, "failure_type", lambda: None)()
            if failure_type is not None:
                return EngineReadiness(
                    False,
                    f"{topology} transport failed: {failure_type}",
                    fatal=True,
                )
            dead = launcher.dead_ranks()
            if dead:
                return EngineReadiness(
                    False,
                    f"{topology} ranks not running: {sorted(dead)}",
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
        if self._shutdown_started:
            return
        task = self._pump_task
        if task is not None and not task.done():
            return
        # EngineLoop.has_work() includes scheduler and in-flight pipeline state.
        # Read it only after the previous worker has settled; producers otherwise
        # race a live step thread for scheduler-owned containers.
        if not self._loop.has_work():
            return
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
        if self._shutdown_started:
            raise RuntimeError("Kairyu backend is shut down")
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

    def _submit(
        self,
        request: GenerationRequest,
        *,
        pre_reserved: bool = False,
        prepared_prompt: PreparedPrompt | None = None,
    ) -> asyncio.Queue:
        request_id = request.request_id
        if not pre_reserved:
            self._reserve_request_ids((request_id,))
        submitted = False
        queue: asyncio.Queue | None = None
        try:
            if prepared_prompt is None:
                prepared_prompt = self._take_or_prepare_request(request)
            self._loop.submit(
                request_id,
                prepared_prompt.prompt,
                request.sampling_params,
                priority=request.priority,
                scheduling_class=request.scheduling_class,
                prepared_prompt=prepared_prompt,
            )
            submitted = True
            queue = asyncio.Queue()
            self._queues[request_id] = queue
            self._ensure_pump()
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
            prompt_token_ids=supplied_prompt_token_ids(request.prompt) or (),
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
            prompt_token_ids=supplied_prompt_token_ids(request.prompt) or (),
        )

    async def _generate_one(
        self,
        request: GenerationRequest,
        *,
        pre_reserved: bool = False,
        prepared_prompt: PreparedPrompt | None = None,
    ) -> GenerationResult:
        queue = self._submit(
            request,
            pre_reserved=pre_reserved,
            prepared_prompt=prepared_prompt,
        )
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
            prepared_prompt = self._take_or_prepare_request(request)
            tasks = [
                asyncio.create_task(
                    self._generate_one(
                        sub,
                        pre_reserved=True,
                        prepared_prompt=prepared_prompt,
                    )
                )
                for sub in subs
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
            prepared_prompt = self._take_or_prepare_request(request)
            for index, sub in enumerate(subs):
                queues[index] = self._submit(
                    sub,
                    pre_reserved=True,
                    prepared_prompt=prepared_prompt,
                )
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

    async def _shutdown_impl(self) -> None:
        pump = self._pump_task
        try:
            try:
                if pump is not None:
                    pump.cancel()
                    try:
                        await pump
                    except asyncio.CancelledError:
                        pass
            finally:
                await asyncio.to_thread(self._loop.close)
        finally:
            with self._prepared_requests_lock:
                self._prepared_requests.clear()
            # Stop spawned distributed ranks even if settling the pump or loop fails.
            launcher = getattr(self._loop, "parallel_launcher", None)
            if launcher is None:
                launcher = getattr(self._loop, "tp_launcher", None)
            if launcher is not None:
                await asyncio.to_thread(launcher.shutdown)

    @staticmethod
    async def _await_shutdown_task(task: asyncio.Task[None]) -> None:
        """Finish the one owned cleanup before propagating caller cancellation."""

        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as cancelled:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
            try:
                task.result()
            except BaseException as cleanup_error:
                raise cancelled from cleanup_error
            raise

    async def shutdown(self) -> None:
        task = self._shutdown_task
        if task is None:
            # Terminal before the first await: no producer may submit work while
            # the shared cleanup task settles the current step and GPU owners.
            self._shutdown_started = True
            task = asyncio.create_task(self._shutdown_impl())
            self._shutdown_task = task
        await self._await_shutdown_task(task)


register_backend("kairyu", KairyuBackend)
