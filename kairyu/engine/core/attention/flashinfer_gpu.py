"""FlashInfer paged-attention adapter (m13 D4) — written locally, GPU-verified
on deploy day (`pytest -m gpu`).

API pins (reviewed against docs.flashinfer.ai 0.6.x — the fake-module contract
tests enforce every one of them):
- both wrappers take one shared, zero-initialized WORKSPACE buffer first
  (394 MiB uint8, matching vLLM's serving default), constructed once — the
  adapter is stateful and must be ONE shared instance across all layers;
- prefill ``plan()`` takes ``head_dim_qk`` (NOT ``head_dim`` — that spelling
  is decode-wrapper-only);
- ``q_data_type``/``kv_data_type`` are passed explicitly (defaults are fp16);
- indptr/indices/last_page_len are int32 tensors — for the list paths indptr
  and last_page_len are built on the HOST and indices on the device; the tensor
  decode path derives all three on the DEVICE (``plan()`` accepts that and does
  its own host copy internally);
- ``causal=True`` is bottom-right aligned: correct iff
  ``chunk_start + T == seq_len`` (asserted; every call site satisfies it);
- the plan is cached per (page_table, seq_len, chunk_start, T) so layer 0
  plans and layers 1..N-1 just run;
- ``plan()`` CANNOT run inside a CUDA graph (FlashInfer's own docstring says
  so, and it copies ``indptr`` to the host to build the split-KV schedule).
  The capture-safe unit is therefore ``run()`` alone, over the persistent
  device buffers a ``use_cuda_graph=True`` wrapper owns — see ``plan_decode``.

This module is coverage-omitted (``*_gpu.py``); its LOGIC is still CPU-tested
via the injected fake module.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from pathlib import Path

import torch

from kairyu.engine.core.kv_pool import PagedKVPool
from kairyu.kernels.flashinfer_decode_plan_gpu import (
    pack_flashinfer_decode_metadata,
)

# FlashInfer calls 128 MiB "recommended", but that is not a capacity guarantee:
# Qwen3-32B's 1,024-token chunked prefill required 190,840,832 bytes once its
# context grew during the 20-item LiveCodeBench gate. vLLM uses 394 MiB for the
# same shared prefill/decode workspace. Sharing keeps the per-rank reservation
# at 394 MiB rather than allocating one buffer per wrapper.
_WORKSPACE_BYTES = 394 * 1024 * 1024


def _ensure_python_bin_on_path() -> None:
    """Expose a venv-local ninja without requiring shell activation.

    FlashInfer invokes ``ninja`` by name only when an exact AOT module is not
    available. Directly running ``.venv/bin/kairyu`` does not put that sibling
    directory on PATH, even though the GPU environment installed the helper.
    """
    path = os.environ.get("PATH", "")
    if shutil.which("ninja", path=path) is not None:
        return
    python_bin = Path(sys.executable).parent
    ninja = python_bin / "ninja"
    if os.access(ninja, os.X_OK):
        os.environ["PATH"] = os.pathsep.join(
            part for part in (str(python_bin), path) if part
        )


def _is_capturing() -> bool:
    """True while the current stream is capturing a CUDA graph.

    Returns False (without creating a CUDA context) on CPU boxes, so the
    capture guards below are free in the eager and CPU-test paths.
    """
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


class FlashInferBackend:
    """One instance serves every layer: shared workspace + plan cache."""

    supports_batched_prefill = True
    supports_fp8_kv = True

    #: ``GraphDecodeBackend`` (see ``kairyu.engine.core.attention``): declared
    #: TRUE only because the host work lives in ``plan_decode`` — outside the
    #: captured region — and ``attend_decode`` below is a bare ``run()`` over
    #: the PERSISTENT paged buffers a ``use_cuda_graph=True`` wrapper owns.
    #: This attribute is what ``PagedModelRunner`` gates on, so it is a promise
    #: about ``attend_decode``, not about this class in general: ``plan()``
    #: still cannot be captured and the eager list paths still sync to the host.
    supports_graph_capture = True
    supports_fast_replay_plan = True

    def __init__(self, device: object = "cuda") -> None:
        _ensure_python_bin_on_path()
        try:
            import flashinfer  # deferred: not installable on macOS; [gpu] extra
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "FlashInfer is required for the flashinfer attention backend "
                "and for FlashAttention delegated decode; run "
                "`uv sync --extra gpu`"
            ) from error

        self._flashinfer = flashinfer
        try:
            decode_module = importlib.import_module("flashinfer.decode")
        except (ImportError, AttributeError):
            self._fast_decode_plan = None
        else:
            candidate = getattr(decode_module, "fast_decode_plan", None)
            self._fast_decode_plan = candidate if callable(candidate) else None
        selected = torch.device(device)
        if selected.type == "cuda" and selected.index is None:
            selected = torch.device("cuda", torch.cuda.current_device())
        self._device = selected
        workspace = torch.zeros(_WORKSPACE_BYTES, dtype=torch.uint8, device=selected)
        self._workspace = workspace
        self._prefill = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
        self._decode = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace, kv_layout="NHD", use_tensor_cores=True
        )
        self._plan_key: tuple | None = None
        self._planned_decode = False
        # tensor decode path: one cudagraph-mode wrapper per (batch, pages)
        # shape. Never replaced once built — a captured graph holds pointers
        # into ITS buffers, so swapping the wrapper would silently make every
        # later replay read a dead plan.
        self._graph_decode: dict[tuple[int, int], object] = {}
        self._graph_decode_initialized: dict[tuple[int, int], tuple] = {}
        self._decode_tensor_wrapper: object | None = None
        self._decode_tensor_key: tuple | None = None
        self._decode_tensor_layers: set[int] = set()
        self._prefill_plan_calls = 0
        self._prefill_run_calls = 0
        self._decode_plan_calls = 0
        self._decode_run_calls = 0
        self._decode_fast_replay_plan_calls = 0
        self._decode_stock_replay_fallback_calls = 0
        self._decode_replay_fallback_reason: str | None = None

    def preflight_runtime(
        self,
        config: object,
        kv_pool: PagedKVPool,
        *,
        q_dtype: torch.dtype,
    ) -> None:
        """Resolve exact prefill/decode modules before readiness is published.

        ``plan`` loads a matching AOT module when present and otherwise performs
        FlashInfer's JIT build.  Running both plans with the live TP-local model
        heads, KV dtype, head width, and page size prevents the first admitted
        request from discovering a missing build helper or unsupported kernel.
        No KV data is read or written.
        """
        num_qo_heads = getattr(config, "num_attention_heads", None)
        if not isinstance(num_qo_heads, int) or num_qo_heads < 1:
            raise RuntimeError(
                "FlashInfer runtime preflight requires a positive "
                "model num_attention_heads"
            )
        qo_indptr = torch.tensor([0, 1], dtype=torch.int32)
        paged_kv_indptr = torch.tensor([0, 1], dtype=torch.int32)
        paged_kv_indices = torch.tensor(
            [0], dtype=torch.int32, device=self._device
        )
        paged_kv_last_page_len = torch.tensor([1], dtype=torch.int32)
        try:
            self._prefill.plan(
                qo_indptr,
                paged_kv_indptr,
                paged_kv_indices,
                paged_kv_last_page_len,
                num_qo_heads,
                kv_pool.num_kv_heads,
                head_dim_qk=kv_pool.head_dim,
                page_size=kv_pool.page_size,
                causal=True,
                q_data_type=q_dtype,
                kv_data_type=kv_pool.k.dtype,
            )
            self._decode.plan(
                paged_kv_indptr,
                paged_kv_indices,
                paged_kv_last_page_len,
                num_qo_heads,
                kv_pool.num_kv_heads,
                kv_pool.head_dim,
                kv_pool.page_size,
                q_data_type=q_dtype,
                kv_data_type=kv_pool.k.dtype,
            )
        except Exception as error:
            raise RuntimeError(
                "FlashInfer runtime preflight failed before serving became "
                f"ready: {type(error).__name__}: {error}"
            ) from error

    @property
    def device(self) -> torch.device:
        """The explicit device owning this stateful backend's workspace."""
        return self._device

    def prefill_execution_stats(self, *, reset: bool = False) -> dict[str, object]:
        """Native prefill plan/run counts for matched structural evidence."""
        result = {
            "type": type(self).__name__,
            "plans": self._prefill_plan_calls,
            "runs": self._prefill_run_calls,
        }
        if reset:
            self._prefill_plan_calls = 0
            self._prefill_run_calls = 0
        return result

    def decode_execution_stats(self, *, reset: bool = False) -> dict[str, object]:
        """Native decode plan/run counts for structural verification evidence."""
        result = {
            "type": type(self).__name__,
            "plans": self._decode_plan_calls,
            "runs": self._decode_run_calls,
            "fast_replay_plans": self._decode_fast_replay_plan_calls,
            "stock_replay_fallbacks": self._decode_stock_replay_fallback_calls,
            "replay_fallback_reason": self._decode_replay_fallback_reason,
        }
        if reset:
            self._decode_plan_calls = 0
            self._decode_run_calls = 0
            self._decode_fast_replay_plan_calls = 0
            self._decode_stock_replay_fallback_calls = 0
            self._decode_replay_fallback_reason = None
        return result

    @staticmethod
    def _kv_scale_kwargs(
        kv_pool: PagedKVPool,
        layer: int,
    ) -> dict[str, float | None]:
        """The cache's explicit calibration contract for one layer."""
        return {
            "k_scale": kv_pool.k_scale(layer),
            "v_scale": kv_pool.v_scale(layer),
        }

    def _paged_arrays(
        self, page_table: list[int], seq_len: int, page_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_pages = -(-seq_len // page_size)
        indptr = torch.tensor([0, num_pages], dtype=torch.int32)  # host
        indices = torch.tensor(page_table[:num_pages], dtype=torch.int32, device=self._device)
        last_page_len = torch.tensor([(seq_len - 1) % page_size + 1], dtype=torch.int32)  # host
        return indptr, indices, last_page_len

    def _paged_batch_arrays(
        self,
        page_tables: list[list[int]],
        seq_lens: list[int],
        page_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        page_counts = [-(-seq_len // page_size) for seq_len in seq_lens]
        offsets = [0]
        indices: list[int] = []
        for page_table, page_count in zip(page_tables, page_counts, strict=True):
            offsets.append(offsets[-1] + page_count)
            indices.extend(page_table[:page_count])
        return (
            torch.tensor(offsets, dtype=torch.int32),
            torch.tensor(indices, dtype=torch.int32, device=self._device),
            torch.tensor(
                [(seq_len - 1) % page_size + 1 for seq_len in seq_lens],
                dtype=torch.int32,
            ),
        )

    def _plan(
        self,
        query: torch.Tensor,
        kv_pool: PagedKVPool,
        page_table: list[int],
        seq_len: int,
        chunk_start: int,
    ) -> bool:
        """Plan (or reuse the cached plan); returns True for the decode path."""
        chunk_len = query.shape[0]
        assert chunk_start + chunk_len == seq_len, (
            "FlashInfer causal=True is bottom-right aligned: the chunk must be "
            f"the tail of the sequence (chunk_start={chunk_start} T={chunk_len} "
            f"seq_len={seq_len})"
        )
        key = (
            tuple(page_table),
            seq_len,
            chunk_start,
            chunk_len,
            query.dtype,
            kv_pool.k.dtype,
            kv_pool.v.dtype,
        )
        is_decode = chunk_len == 1
        if key == self._plan_key and is_decode == self._planned_decode:
            return is_decode
        indptr, indices, last_page_len = self._paged_arrays(page_table, seq_len, kv_pool.page_size)
        if is_decode:
            self._decode.plan(
                indptr,
                indices,
                last_page_len,
                query.shape[1],  # num_qo_heads
                kv_pool.num_kv_heads,
                kv_pool.head_dim,
                kv_pool.page_size,
                q_data_type=query.dtype,
                kv_data_type=kv_pool.k.dtype,
            )
            self._decode_plan_calls += 1
        else:
            qo_indptr = torch.tensor([0, chunk_len], dtype=torch.int32)  # host
            self._prefill.plan(
                qo_indptr,
                indptr,
                indices,
                last_page_len,
                query.shape[1],  # num_qo_heads
                kv_pool.num_kv_heads,
                head_dim_qk=kv_pool.head_dim,  # NOT head_dim (prefill spelling)
                page_size=kv_pool.page_size,
                causal=True,
                q_data_type=query.dtype,
                kv_data_type=kv_pool.k.dtype,
            )
            self._prefill_plan_calls += 1
        self._plan_key = key
        self._planned_decode = is_decode
        return is_decode

    def attend(
        self,
        query: torch.Tensor,
        kv_pool: PagedKVPool,
        layer: int,
        page_table: list[int],
        seq_len: int,
        chunk_start: int,
    ) -> torch.Tensor:
        is_decode = self._plan(query, kv_pool, page_table, seq_len, chunk_start)
        paged_kv = (kv_pool.k[layer], kv_pool.v[layer])  # NHD tuple form
        wrapper = self._decode if is_decode else self._prefill
        if is_decode:
            # flashinfer 0.6.x decode.run expects a BATCHED [B, H, D] query;
            # a 2D [H, D] slice raises IndexError in the tvm_ffi kernel. The
            # single-sequence decode path has B=1, so pass the [1, H, D] query
            # as-is (the batched path in attend_batched already does this).
            out = wrapper.run(
                query,
                paged_kv,
                **self._kv_scale_kwargs(kv_pool, layer),
            )  # decode: [B=1, H, D] query
            self._decode_run_calls += 1
            return out.reshape(1, -1)
        out = wrapper.run(
            query,
            paged_kv,
            **self._kv_scale_kwargs(kv_pool, layer),
        )  # [T, H, D]
        self._prefill_run_calls += 1
        return out.reshape(query.shape[0], -1)

    def _graph_decode_wrapper(self, batch_size: int, max_pages: int) -> object:
        """The ``use_cuda_graph=True`` decode wrapper for this input SHAPE.

        cudagraph mode is what gives the plan somewhere PERSISTENT to live:
        without it ``plan()`` rebinds ``_paged_kv_*_buf`` to freshly allocated
        tensors, so a captured ``run()`` would keep reading the buffers that
        existed at capture time. Batch size is fixed per wrapper (FlashInfer
        requires it), and the indices buffer is sized for the widest page table
        the shape can produce, so re-planning never has to grow it.
        """
        shape = (batch_size, max_pages)
        wrapper = self._graph_decode.get(shape)
        if wrapper is not None:
            return wrapper
        indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=self._device)
        indices = torch.zeros(
            max(batch_size * max_pages, 1), dtype=torch.int32, device=self._device
        )
        last_page_len = torch.ones(batch_size, dtype=torch.int32, device=self._device)
        wrapper = self._flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self._workspace,
            kv_layout="NHD",
            use_cuda_graph=True,
            use_tensor_cores=True,
            paged_kv_indptr_buffer=indptr,
            paged_kv_indices_buffer=indices,
            paged_kv_last_page_len_buffer=last_page_len,
        )
        self._graph_decode[shape] = wrapper
        return wrapper

    @staticmethod
    def _decode_shape_key(
        kv_pool: PagedKVPool,
        page_tables: torch.Tensor,
        num_qo_heads: int,
        q_dtype: torch.dtype,
    ) -> tuple:
        """Identity of a plan, from METADATA only — shapes and dtypes are host
        attributes, so comparing them costs no synchronization."""
        return (
            int(page_tables.shape[0]),
            int(page_tables.shape[1]),
            num_qo_heads,
            kv_pool.num_kv_heads,
            kv_pool.head_dim,
            kv_pool.page_size,
            q_dtype,
            kv_pool.k.dtype,
            kv_pool.v.dtype,
        )

    @staticmethod
    def _wrapper_paged_buffers(
        wrapper: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Resolve the persistent buffers owned by a cudagraph wrapper."""

        names = (
            ("_paged_kv_indptr_buf", "indptr_buffer"),
            ("_paged_kv_indices_buf", "indices_buffer"),
            ("_paged_kv_last_page_len_buf", "last_page_len_buffer"),
        )
        resolved: list[torch.Tensor] = []
        for private, contract_fake in names:
            value = getattr(wrapper, private, None)
            if value is None:
                value = getattr(wrapper, contract_fake, None)
            if not torch.is_tensor(value):
                raise AttributeError(
                    "FlashInfer CUDA-graph wrapper does not expose its "
                    f"persistent {private} buffer"
                )
            resolved.append(value)
        return resolved[0], resolved[1], resolved[2]

    @staticmethod
    def _host_decode_arrays(
        host_seq_lens: tuple[int, ...],
        *,
        page_size: int,
        max_pages: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Build the CPU-only schedule inputs from scheduler-owned lengths."""

        if not host_seq_lens:
            raise ValueError("fast replay planning needs non-empty host_seq_lens")
        if page_size < 1 or max_pages < 1:
            raise ValueError("decode page geometry must be positive")
        page_counts: list[int] = []
        last_lengths: list[int] = []
        for seq_len in host_seq_lens:
            if type(seq_len) is not int or seq_len < 1:
                raise ValueError("host_seq_lens must contain positive integers")
            page_count = (seq_len + page_size - 1) // page_size
            if page_count > max_pages:
                raise ValueError(
                    f"seq_len={seq_len} needs {page_count} pages, but the "
                    f"captured table has width {max_pages}"
                )
            page_counts.append(page_count)
            last_lengths.append((seq_len - 1) % page_size + 1)
        offsets = [0]
        for page_count in page_counts:
            offsets.append(offsets[-1] + page_count)
        return (
            torch.tensor(offsets, dtype=torch.int32, device="cpu"),
            torch.tensor(last_lengths, dtype=torch.int32, device="cpu"),
            offsets[-1],
        )

    def _record_stock_replay_fallback(self, reason: object) -> None:
        self._decode_stock_replay_fallback_calls += 1
        self._decode_replay_fallback_reason = str(reason)

    def _try_fast_replay_plan(
        self,
        kv_pool: PagedKVPool,
        page_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        num_qo_heads: int,
        q_dtype: torch.dtype,
        host_seq_lens: tuple[int, ...] | None,
    ) -> bool:
        """Refresh an initialized wrapper without any CUDA-to-host read.

        CUDA-graph replay and eager tensor decode share this implementation.
        The public stock planner remains the one-time initializer for each
        wrapper shape.
        """

        if self._fast_decode_plan is None:
            self._record_stock_replay_fallback(
                "flashinfer.decode.fast_decode_plan is unavailable"
            )
            return False
        rows, max_pages = int(page_tables.shape[0]), int(page_tables.shape[1])
        if host_seq_lens is None or len(host_seq_lens) != rows:
            self._record_stock_replay_fallback(
                "authoritative host_seq_lens do not match the replay batch"
            )
            return False
        key = self._decode_shape_key(kv_pool, page_tables, num_qo_heads, q_dtype)
        shape = (rows, max_pages)
        wrapper = self._graph_decode.get(shape)
        if (
            wrapper is None
            or self._graph_decode_initialized.get(shape) != key
        ):
            self._record_stock_replay_fallback(
                "the CUDA-graph wrapper has not been initialized by stock plan()"
            )
            return False

        try:
            indptr_cpu, last_page_len_cpu, used_pages = self._host_decode_arrays(
                host_seq_lens,
                page_size=kv_pool.page_size,
                max_pages=max_pages,
            )
            indptr_buf, indices_buf, last_page_len_buf = (
                self._wrapper_paged_buffers(wrapper)
            )
            pack_flashinfer_decode_metadata(
                page_tables,
                seq_lens,
                indptr_buf,
                indices_buf,
                last_page_len_buf,
                page_size=kv_pool.page_size,
            )
            # 0.6.14's fast helper is an unbound function whose first argument
            # is the wrapper.  In cudagraph mode it deliberately does not copy
            # these inputs; the Triton launch above has already refreshed the
            # persistent device buffers on this same current stream.  Both
            # host tensors below are ordinary CPU tensors, so its `.cpu()`
            # calls are no-ops rather than D2H transfers.
            self._fast_decode_plan(
                wrapper,
                indptr_cpu,
                indices_buf[:used_pages],
                last_page_len_cpu,
                num_qo_heads,
                kv_pool.num_kv_heads,
                kv_pool.head_dim,
                kv_pool.page_size,
                q_data_type=q_dtype,
                kv_data_type=kv_pool.k.dtype,
                global_override_indptr_cpu=indptr_cpu,
            )
        except Exception as error:
            # Version/signature/layout incompatibility is an optimization miss,
            # never a correctness mode. This intentionally includes Triton
            # compiler exceptions whose concrete classes vary by release.
            # Stock plan() rewrites every persistent buffer after this attempt
            # and remains the truthful fallback.
            self._record_stock_replay_fallback(
                f"fast replay plan was incompatible: {type(error).__name__}: {error}"
            )
            return False

        self._decode_plan_calls += 1
        self._decode_fast_replay_plan_calls += 1
        self._decode_replay_fallback_reason = None
        self._decode_tensor_wrapper = wrapper
        self._decode_tensor_key = key
        self._decode_tensor_layers = set()
        return True

    def plan_decode(
        self,
        kv_pool: PagedKVPool,
        page_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        num_qo_heads: int,
        q_dtype: torch.dtype,
        replay: bool = False,
        host_seq_lens: tuple[int, ...] | None = None,
    ) -> None:
        """Plan stock initialization or refresh an initialized decode shape.

        Authoritative scheduler-owned host lengths opt eager tensor decode into
        the same no-D2H fast planner used by graph replay.  The first occurrence
        of a shape still uses stock ``plan()`` to initialize FlashInfer's module
        and persistent wrapper buffers.
        """

        if _is_capturing():
            raise RuntimeError(
                "FlashInfer plan() cannot run inside a CUDA graph; call "
                "plan_decode() once per decode step BEFORE capture/replay"
            )
        if (replay or host_seq_lens is not None) and self._try_fast_replay_plan(
            kv_pool,
            page_tables,
            seq_lens,
            num_qo_heads=num_qo_heads,
            q_dtype=q_dtype,
            host_seq_lens=host_seq_lens,
        ):
            return
        self._plan_decode_stock(
            kv_pool,
            page_tables,
            seq_lens,
            num_qo_heads=num_qo_heads,
            q_dtype=q_dtype,
        )

    def _plan_decode_stock(
        self,
        kv_pool: PagedKVPool,
        page_tables: torch.Tensor,  # [B, P] int
        seq_lens: torch.Tensor,  # [B] int
        *,
        num_qo_heads: int,
        q_dtype: torch.dtype,
    ) -> None:
        """HOST phase of the tensor decode path — run it ONCE PER STEP, and
        never inside a CUDA graph.

        FlashInfer's ``plan()`` builds the split-KV schedule on the CPU (it
        copies ``indptr`` to the host itself) and its docs state outright that
        it "cannot be used in Cuda Graph". That is a kernel-API limit, not a
        choice: the capture-safe decomposition is plan-outside / run-inside,
        which is exactly what the ``use_cuda_graph`` buffers exist for. Calling
        this before each replay refreshes the schedule and the device-side
        indptr/indices/last_page_len the captured ``run()`` already points at,
        so the graph attends over the CURRENT step's pages.

        The paged arrays themselves are derived on DEVICE from the page-table
        and length tensors.  FlashInfer's stock plan then copies their schedule
        inputs to the host; graph replay avoids that via the separate fast path.
        """
        page_size = kv_pool.page_size
        pages_per_row = torch.div(seq_lens + page_size - 1, page_size, rounding_mode="floor").to(
            torch.int32
        )
        indptr = torch.zeros(
            pages_per_row.shape[0] + 1, dtype=torch.int32, device=pages_per_row.device
        )
        indptr[1:] = torch.cumsum(pages_per_row, dim=0)
        # only the pages each row actually uses, concatenated in row order
        span = torch.arange(page_tables.shape[1], device=page_tables.device)
        used = span[None, :] < pages_per_row[:, None]
        indices = page_tables[used].to(torch.int32)
        last_page_len = ((seq_lens - 1) % page_size + 1).to(torch.int32)

        wrapper = self._graph_decode_wrapper(int(page_tables.shape[0]), int(page_tables.shape[1]))
        wrapper.plan(
            indptr,
            indices,
            last_page_len,
            num_qo_heads,
            kv_pool.num_kv_heads,
            kv_pool.head_dim,
            page_size,
            q_data_type=q_dtype,
            kv_data_type=kv_pool.k.dtype,
        )
        self._decode_plan_calls += 1
        self._decode_tensor_wrapper = wrapper
        self._decode_tensor_key = self._decode_shape_key(
            kv_pool, page_tables, num_qo_heads, q_dtype
        )
        self._graph_decode_initialized[
            (int(page_tables.shape[0]), int(page_tables.shape[1]))
        ] = self._decode_tensor_key
        self._decode_tensor_layers = set()

    def attend_decode(
        self,
        query: torch.Tensor,  # [B, heads, head_dim]
        kv_pool: PagedKVPool,
        layer: int,
        page_tables: torch.Tensor,  # [B, P] int
        seq_lens: torch.Tensor,  # [B] int
    ) -> torch.Tensor:
        """Tensor-input decode, the same contract the torch backend implements.

        This call is the CAPTURE-SAFE half: it is one ``run()`` over buffers
        that already hold the plan — no ``.tolist()``, no ``.cpu()``, no
        ``plan()``, nothing that synchronizes with the host. Inside a capture
        the step's ``plan_decode`` must therefore already have happened, and if
        it has not this raises instead of silently baking capture-time pages
        into the graph.

        Eagerly nobody has to know that: the plan is refreshed lazily, once per
        step (detected by a layer being revisited), so this stays a drop-in for
        ``TorchAttentionBackend.attend_decode``.
        """
        key = self._decode_shape_key(kv_pool, page_tables, int(query.shape[1]), query.dtype)
        if _is_capturing():
            if self._decode_tensor_key != key:
                raise RuntimeError(
                    "FlashInfer decode has no live plan for this shape and "
                    "plan() cannot run inside a CUDA graph: call plan_decode() "
                    "for this step before capturing/replaying the graph"
                )
        elif self._decode_tensor_key != key or layer in self._decode_tensor_layers:
            # a repeated layer means a NEW step over the same buffers, whose
            # contents this adapter must not read to find out (that is the host
            # sync); re-planning is both correct and once-per-step cheap
            self.plan_decode(
                kv_pool,
                page_tables,
                seq_lens,
                num_qo_heads=int(query.shape[1]),
                q_dtype=query.dtype,
            )
        self._decode_tensor_layers.add(layer)
        out = self._decode_tensor_wrapper.run(  # type: ignore[union-attr]
            query,
            (kv_pool.k[layer], kv_pool.v[layer]),
            **self._kv_scale_kwargs(kv_pool, layer),
        )
        self._decode_run_calls += 1
        return out.reshape(query.shape[0], -1)

    def attend_batched(
        self,
        queries: list[torch.Tensor],
        kv_pool: PagedKVPool,
        layer: int,
        page_tables: list[list[int]],
        seq_lens: list[int],
        chunk_starts: list[int],
    ) -> list[torch.Tensor]:
        lengths = {
            "queries": len(queries),
            "page_tables": len(page_tables),
            "seq_lens": len(seq_lens),
            "chunk_starts": len(chunk_starts),
        }
        if len(set(lengths.values())) != 1:
            details = ", ".join(f"{name}={length}" for name, length in lengths.items())
            raise ValueError(
                f"FlashInfer parallel batch inputs must have the same length ({details})"
            )
        if not queries:
            return []
        for query, seq_len, chunk_start in zip(queries, seq_lens, chunk_starts, strict=True):
            chunk_len = query.shape[0]
            if chunk_len < 1:
                raise ValueError("FlashInfer batched attention rows must not be empty")
            assert chunk_start + chunk_len == seq_len, (
                "FlashInfer causal=True is bottom-right aligned: the chunk must be "
                f"the tail of the sequence (chunk_start={chunk_start} T={chunk_len} "
                f"seq_len={seq_len})"
            )
        first = queries[0]
        if any(
            query.device != first.device
            or query.dtype != first.dtype
            or query.shape[1:] != first.shape[1:]
            for query in queries[1:]
        ):
            raise ValueError(
                "FlashInfer batched attention queries must share device, dtype, "
                "head count, and head dimension"
            )

        query_lens = tuple(int(query.shape[0]) for query in queries)
        is_decode = all(query_len == 1 for query_len in query_lens)
        if not is_decode:
            offsets = [0]
            for query_len in query_lens:
                offsets.append(offsets[-1] + query_len)
            flat = self.attend_prefill(
                torch.cat(queries, dim=0),
                kv_pool,
                layer,
                tuple(tuple(page_table) for page_table in page_tables),
                tuple(seq_lens),
                tuple(chunk_starts),
                tuple(offsets),
            )
            return list(torch.split(flat, query_lens, dim=0))

        key = (
            "batched_decode",
            tuple(tuple(page_table) for page_table in page_tables),
            tuple(seq_lens),
            tuple(chunk_starts),
            query_lens,
            first.dtype,
            kv_pool.k.dtype,
            kv_pool.v.dtype,
        )
        if key != self._plan_key or not self._planned_decode:
            indptr, indices, last_page_len = self._paged_batch_arrays(
                page_tables, seq_lens, kv_pool.page_size
            )
            self._decode.plan(
                indptr,
                indices,
                last_page_len,
                first.shape[1],
                kv_pool.num_kv_heads,
                kv_pool.head_dim,
                kv_pool.page_size,
                q_data_type=first.dtype,
                kv_data_type=kv_pool.k.dtype,
            )
            self._decode_plan_calls += 1
            self._plan_key = key
            self._planned_decode = True

        query_batch = torch.cat(queries, dim=0)
        paged_kv = (kv_pool.k[layer], kv_pool.v[layer])
        out = self._decode.run(
            query_batch,
            paged_kv,
            **self._kv_scale_kwargs(kv_pool, layer),
        )
        self._decode_run_calls += 1
        contexts = out.reshape(query_batch.shape[0], -1)
        return list(torch.split(contexts, query_lens, dim=0))

    def attend_prefill(
        self,
        query: torch.Tensor,
        kv_pool: PagedKVPool,
        layer: int,
        page_tables: tuple[tuple[int, ...], ...],
        seq_lens: tuple[int, ...],
        chunk_starts: tuple[int, ...],
        qo_indptr: tuple[int, ...],
    ) -> torch.Tensor:
        """One FlashInfer ragged-prefill plan shared by every model layer."""
        batch = len(seq_lens)
        lengths = {
            "page_tables": len(page_tables),
            "seq_lens": batch,
            "chunk_starts": len(chunk_starts),
            "qo_indptr": len(qo_indptr) - 1,
        }
        if batch < 1 or len(set(lengths.values())) != 1:
            details = ", ".join(f"{name}={length}" for name, length in lengths.items())
            raise ValueError(
                "FlashInfer ragged prefill inputs must describe the same "
                f"non-empty batch ({details})"
            )
        if qo_indptr[0] != 0 or qo_indptr[-1] != query.shape[0]:
            raise ValueError("FlashInfer qo_indptr must span the flat query exactly")
        query_lens = tuple(qo_indptr[index + 1] - qo_indptr[index] for index in range(batch))
        if any(query_len < 1 for query_len in query_lens):
            raise ValueError("FlashInfer ragged prefill rows must not be empty")
        for query_len, seq_len, chunk_start in zip(query_lens, seq_lens, chunk_starts, strict=True):
            assert chunk_start + query_len == seq_len, (
                "FlashInfer causal=True is bottom-right aligned: the chunk must "
                "be the tail of the sequence "
                f"(chunk_start={chunk_start} T={query_len} seq_len={seq_len})"
            )

        key = (
            "batched_prefill",
            page_tables,
            seq_lens,
            chunk_starts,
            query_lens,
            query.dtype,
            kv_pool.k.dtype,
            kv_pool.v.dtype,
        )
        if key != self._plan_key or self._planned_decode:
            kv_indptr, kv_indices, last_page_len = self._paged_batch_arrays(
                page_tables, seq_lens, kv_pool.page_size
            )
            self._prefill.plan(
                torch.tensor(qo_indptr, dtype=torch.int32),
                kv_indptr,
                kv_indices,
                last_page_len,
                query.shape[1],
                kv_pool.num_kv_heads,
                head_dim_qk=kv_pool.head_dim,
                page_size=kv_pool.page_size,
                causal=True,
                q_data_type=query.dtype,
                kv_data_type=kv_pool.k.dtype,
            )
            self._prefill_plan_calls += 1
            self._plan_key = key
            self._planned_decode = False
        out = self._prefill.run(
            query,
            (kv_pool.k[layer], kv_pool.v[layer]),
            **self._kv_scale_kwargs(kv_pool, layer),
        )
        self._prefill_run_calls += 1
        return out.reshape(query.shape[0], -1)
