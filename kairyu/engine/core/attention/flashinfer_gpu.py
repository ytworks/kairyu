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
- indptr/indices/last_page_len are int32 tensors — indptr and last_page_len on
  HOST, indices on the device;
- ``causal=True`` is bottom-right aligned: correct iff
  ``chunk_start + T == seq_len`` (asserted; every call site satisfies it);
- the plan is cached per (page_table, seq_len, chunk_start, T) so layer 0
  plans and layers 1..N-1 just run.

This module is coverage-omitted (``*_gpu.py``); its LOGIC is still CPU-tested
via the injected fake module.
"""

from __future__ import annotations

import torch

from kairyu.engine.core.kv_pool import PagedKVPool

_WORKSPACE_BYTES = 128 * 1024 * 1024


class FlashInferBackend:
    """One instance serves every layer: shared workspace + plan cache."""

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
        self._decode = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            decode_workspace, kv_layout="NHD", use_tensor_cores=True
        )
        self._plan_key: tuple | None = None
        self._planned_decode = False

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
            out = wrapper.run(query[0], paged_kv)  # decode: [H, D] query
            return out.reshape(1, -1)
        out = wrapper.run(query, paged_kv)  # [T, H, D]
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
