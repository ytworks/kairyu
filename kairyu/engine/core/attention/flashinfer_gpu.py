"""FlashInfer paged-attention adapter (m13 D4) — written locally, GPU-verified
on deploy day (`pytest -m gpu`).

API pins (reviewed against docs.flashinfer.ai 0.6.x — the fake-module contract
tests enforce every one of them):
- both wrappers take a zero-initialized WORKSPACE buffer first (128 MB uint8),
  constructed once — the adapter is stateful and must be ONE shared instance
  across all layers;
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

import torch

from kairyu.engine.core.kv_pool import PagedKVPool

_WORKSPACE_BYTES = 128 * 1024 * 1024


def _is_capturing() -> bool:
    """True while the current stream is capturing a CUDA graph.

    Returns False (without creating a CUDA context) on CPU boxes, so the
    capture guards below are free in the eager and CPU-test paths.
    """
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


class FlashInferBackend:
    """One instance serves every layer: shared workspace + plan cache."""

    #: ``GraphDecodeBackend`` (see ``kairyu.engine.core.attention``): declared
    #: TRUE only because the host work lives in ``plan_decode`` — outside the
    #: captured region — and ``attend_decode`` below is a bare ``run()`` over
    #: the PERSISTENT paged buffers a ``use_cuda_graph=True`` wrapper owns.
    #: This attribute is what ``PagedModelRunner`` gates on, so it is a promise
    #: about ``attend_decode``, not about this class in general: ``plan()``
    #: still cannot be captured and the eager list paths still sync to the host.
    supports_graph_capture = True

    def __init__(self, device: str = "cuda") -> None:
        import flashinfer  # deferred: not installable on macOS; [gpu] extra

        self._flashinfer = flashinfer
        self._device = device
        workspace = torch.zeros(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        self._prefill = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, kv_layout="NHD"
        )
        decode_workspace = torch.zeros(
            _WORKSPACE_BYTES, dtype=torch.uint8, device=device
        )
        self._decode_workspace = decode_workspace
        self._decode = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            decode_workspace, kv_layout="NHD", use_tensor_cores=True
        )
        self._plan_key: tuple | None = None
        self._planned_decode = False
        # tensor decode path: one cudagraph-mode wrapper per (batch, pages)
        # shape. Never replaced once built — a captured graph holds pointers
        # into ITS buffers, so swapping the wrapper would silently make every
        # later replay read a dead plan.
        self._graph_decode: dict[tuple[int, int], object] = {}
        self._decode_tensor_wrapper: object | None = None
        self._decode_tensor_key: tuple | None = None
        self._decode_tensor_layers: set[int] = set()

    def _paged_arrays(
        self, page_table: list[int], seq_len: int, page_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_pages = -(-seq_len // page_size)
        indptr = torch.tensor([0, num_pages], dtype=torch.int32)  # host
        indices = torch.tensor(
            page_table[:num_pages], dtype=torch.int32, device=self._device
        )
        last_page_len = torch.tensor(
            [(seq_len - 1) % page_size + 1], dtype=torch.int32
        )  # host
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
        key = (tuple(page_table), seq_len, chunk_start, chunk_len)
        is_decode = chunk_len == 1
        if key == self._plan_key and is_decode == self._planned_decode:
            return is_decode
        indptr, indices, last_page_len = self._paged_arrays(
            page_table, seq_len, kv_pool.page_size
        )
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
            out = wrapper.run(query, paged_kv)  # decode: [B=1, H, D] query
            return out.reshape(1, -1)
        out = wrapper.run(query, paged_kv)  # [T, H, D]
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
            self._decode_workspace,
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
        )

    def plan_decode(
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
        and length tensors: no Python list and no D2H copy in this adapter.
        """
        if _is_capturing():
            raise RuntimeError(
                "FlashInfer plan() cannot run inside a CUDA graph; call "
                "plan_decode() once per decode step BEFORE capture/replay"
            )
        page_size = kv_pool.page_size
        pages_per_row = torch.div(
            seq_lens + page_size - 1, page_size, rounding_mode="floor"
        ).to(torch.int32)
        indptr = torch.zeros(
            pages_per_row.shape[0] + 1, dtype=torch.int32, device=pages_per_row.device
        )
        indptr[1:] = torch.cumsum(pages_per_row, dim=0)
        # only the pages each row actually uses, concatenated in row order
        span = torch.arange(page_tables.shape[1], device=page_tables.device)
        used = span[None, :] < pages_per_row[:, None]
        indices = page_tables[used].to(torch.int32)
        last_page_len = ((seq_lens - 1) % page_size + 1).to(torch.int32)

        wrapper = self._graph_decode_wrapper(
            int(page_tables.shape[0]), int(page_tables.shape[1])
        )
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
        self._decode_tensor_wrapper = wrapper
        self._decode_tensor_key = self._decode_shape_key(
            kv_pool, page_tables, num_qo_heads, q_dtype
        )
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
        key = self._decode_shape_key(
            kv_pool, page_tables, int(query.shape[1]), query.dtype
        )
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
            query, (kv_pool.k[layer], kv_pool.v[layer])
        )
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
                "FlashInfer parallel batch inputs must have the same length "
                f"({details})"
            )
        if not queries:
            return []
        if any(query.shape[0] != 1 for query in queries):
            return [
                self.attend(
                    query,
                    kv_pool,
                    layer,
                    page_table,
                    seq_len,
                    chunk_start,
                )
                for query, page_table, seq_len, chunk_start in zip(
                    queries, page_tables, seq_lens, chunk_starts, strict=True
                )
            ]

        for query, seq_len, chunk_start in zip(
            queries, seq_lens, chunk_starts, strict=True
        ):
            chunk_len = query.shape[0]
            assert chunk_start + chunk_len == seq_len, (
                "FlashInfer causal=True is bottom-right aligned: the chunk must be "
                f"the tail of the sequence (chunk_start={chunk_start} T={chunk_len} "
                f"seq_len={seq_len})"
            )

        key = (
            "batched_decode",
            tuple(tuple(page_table) for page_table in page_tables),
            tuple(seq_lens),
            tuple(chunk_starts),
        )
        if key != self._plan_key or not self._planned_decode:
            indptr, indices, last_page_len = self._paged_batch_arrays(
                page_tables, seq_lens, kv_pool.page_size
            )
            self._decode.plan(
                indptr,
                indices,
                last_page_len,
                queries[0].shape[1],
                kv_pool.num_kv_heads,
                kv_pool.head_dim,
                kv_pool.page_size,
                q_data_type=queries[0].dtype,
                kv_data_type=kv_pool.k.dtype,
            )
            self._plan_key = key
            self._planned_decode = True

        query_batch = torch.cat(queries, dim=0)
        paged_kv = (kv_pool.k[layer], kv_pool.v[layer])
        out = self._decode.run(query_batch, paged_kv)
        return [row.reshape(1, -1) for row in out.unbind(dim=0)]
