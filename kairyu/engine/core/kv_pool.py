"""PagedKVPool: multi-layer KV tensors indexed by the scheduler's page ids (m12 D3).

Layout is layer-major — ``[num_layers, num_pages, page_size, num_kv_heads,
head_dim]`` per K and V — so M18's KVTransport fragments (per-layer ×
per-shard) slice contiguously. The same page ids index every layer;
``RadixKVCache`` and the ``Scheduler`` stay page-id accounting only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.models.config import ModelConfig


class PagedKVPool:
    def __init__(
        self,
        num_layers: int,
        num_pages: int,
        page_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        v_head_dim: int | None = None,
    ) -> None:
        # MLA caches only the latent in k (v width 0) — m15 A7
        v_dim = head_dim if v_head_dim is None else v_head_dim
        self.k = torch.zeros(
            (num_layers, num_pages, page_size, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        self.v = torch.zeros(
            (num_layers, num_pages, page_size, num_kv_heads, v_dim),
            dtype=dtype,
            device=device,
        )
        self.num_layers = num_layers
        self.num_pages = num_pages
        self.page_size = page_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.v_head_dim = v_dim

    @staticmethod
    def _cache_payload(
        payload: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert one projected K/V payload to its storage representation.

        E4M3 KV uses an explicit unit-scale quantizer.  Relying on indexed
        assignment to perform the conversion is not portable: PyTorch rejects
        BF16-to-FP8 ``index_put`` on both CPU and CUDA.  Direct writes and the
        torch fallbacks convert here; the fused Triton slot mappers implement
        the same SATFINITE boundary while storing, avoiding a separate launch.
        """
        if dtype == torch.float8_e4m3fn and payload.dtype != dtype:
            # Match NVIDIA's SATFINITE E4M3 conversion contract.  PyTorch's
            # plain cast emits NaN for sufficiently large finite inputs instead
            # of saturating them, which would poison every later attention
            # result that reads the affected cache slot.
            limit = torch.finfo(dtype).max
            return payload.clamp(min=-limit, max=limit).to(dtype=dtype)
        return payload

    def _layer_scale(self, layer: int, dtype: torch.dtype) -> float | None:
        if not 0 <= layer < self.num_layers:
            raise IndexError(
                f"KV layer {layer} is outside [0, {self.num_layers})"
            )
        return 1.0 if dtype == torch.float8_e4m3fn else None

    def k_scale(self, layer: int) -> float | None:
        """Unit calibration scale for an FP8 K cache, otherwise ``None``."""
        return self._layer_scale(layer, self.k.dtype)

    def v_scale(self, layer: int) -> float | None:
        """Unit calibration scale for an FP8 V cache, otherwise ``None``."""
        return self._layer_scale(layer, self.v.dtype)

    @classmethod
    def for_cache(
        cls,
        cache: RadixKVCache,
        config: ModelConfig,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> PagedKVPool:
        """Single source of truth for sizing (m12 D3): pool follows the cache."""
        return cls(
            num_layers=config.num_hidden_layers,
            num_pages=cache.num_pages,
            page_size=cache.page_size,
            num_kv_heads=config.kv_cache_num_heads,
            head_dim=config.kv_cache_head_dim,
            dtype=dtype,
            device=device,
            v_head_dim=config.kv_cache_v_head_dim,
        )

    @property
    def bytes_per_token(self) -> int:
        element = self.k.element_size()
        per_head = self.head_dim + self.v_head_dim
        return self.num_layers * self.num_kv_heads * per_head * element

    def _flat_indices(self, page_table: list[int], positions: torch.Tensor) -> torch.Tensor:
        # index tensors must live on the pool's device: on GPU `positions` arrives
        # on-device, so a CPU page-table tensor would raise on the gather. CPU pools
        # keep the original behaviour (self.k.device == cpu).
        table = torch.tensor(page_table, dtype=torch.long, device=self.k.device)
        pages = table[positions // self.page_size]
        return pages * self.page_size + positions % self.page_size

    def write(
        self,
        layer: int,
        page_table: list[int],
        positions: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """keys/values: [T, num_kv_heads, head_dim] at absolute positions [T]."""
        flat = self._flat_indices(page_table, positions)
        keys = self._cache_payload(keys, dtype=self.k.dtype)
        values = self._cache_payload(values, dtype=self.v.dtype)
        self.k[layer].reshape(-1, self.num_kv_heads, self.head_dim)[flat] = keys
        if self.v_head_dim:  # MLA pools have no v payload (width 0)
            self.v[layer].reshape(-1, self.num_kv_heads, self.v_head_dim)[flat] = values

    def write_batched(
        self,
        layer: int,
        page_tables: torch.Tensor,  # [B, P] int, one row per sequence
        positions: torch.Tensor,  # [B] absolute position of each new token
        keys: torch.Tensor,  # [B, num_kv_heads, head_dim]
        values: torch.Tensor,
        write_from: torch.Tensor | None = None,  # [B], retain KV below this
    ) -> None:
        """Batched single-token write, entirely on device.

        `write()` takes a Python page list and so needs the page ids on the host;
        a captured CUDA graph cannot see a Python list change between replays.
        Here the slot is computed from the page-table TENSOR, so the same kernels
        write wherever the buffers currently point (m17 D1's static-buffer rule).
        """
        if self.k.device.type == "cuda" and self.v_head_dim == self.head_dim:
            # Import only on the CUDA path: CPU-only installs retain the plain
            # torch implementation and do not need Triton to import Kairyu.
            from kairyu.kernels.paged_kv_write_gpu import try_write_batched

            if try_write_batched(
                self.k[layer],
                self.v[layer],
                self.page_size,
                page_tables,
                positions,
                keys,
                values,
                write_from,
            ):
                return
        keys = self._cache_payload(keys, dtype=self.k.dtype)
        values = self._cache_payload(values, dtype=self.v.dtype)
        page_index = torch.div(positions, self.page_size, rounding_mode="floor")
        slot = positions - page_index * self.page_size
        pages = page_tables.gather(1, page_index.unsqueeze(1).long()).squeeze(1)
        flat = pages.long() * self.page_size + slot.long()
        k_flat = self.k[layer].reshape(-1, self.num_kv_heads, self.head_dim)
        if write_from is not None:
            writable = (positions >= write_from)[:, None, None]
            keys = torch.where(writable, keys, k_flat[flat])
        k_flat[flat] = keys
        if self.v_head_dim:
            v_flat = self.v[layer].reshape(
                -1, self.num_kv_heads, self.v_head_dim
            )
            if write_from is not None:
                values = torch.where(writable, values, v_flat[flat])
            v_flat[flat] = values

    def write_ragged(
        self,
        layer: int,
        page_tables: torch.Tensor,  # [B, P] int
        row_ids: torch.Tensor,  # [T] owner row for each flat token
        positions: torch.Tensor,  # [T] absolute position within its owner
        keys: torch.Tensor,  # [T, num_kv_heads, head_dim]
        values: torch.Tensor,
        write_from: torch.Tensor,  # [B], retain cached KV below this
    ) -> None:
        """Write a variable-token prefill batch without crossing row ownership.

        ``build_prefill_batch`` validates that writable physical slots are
        unique across rows.  This device half derives every destination from
        ``(row_ids, positions, page_tables)`` and filters cached positions
        before assignment, so shared radix pages are never rewritten.
        """
        if row_ids.ndim != 1 or positions.ndim != 1:
            raise ValueError("ragged row_ids and positions must be one-dimensional")
        if row_ids.shape != positions.shape or positions.shape[0] != keys.shape[0]:
            raise ValueError(
                "ragged KV inputs must have one row id and position per token"
            )
        if values.shape[0] != keys.shape[0]:
            raise ValueError("ragged keys and values must have the same token count")
        if write_from.ndim != 1 or write_from.shape[0] != page_tables.shape[0]:
            raise ValueError("write_from must have one entry per ragged row")

        if self.k.device.type == "cuda" and self.v_head_dim == self.head_dim:
            from kairyu.kernels.paged_kv_write_gpu import try_write_ragged

            if try_write_ragged(
                self.k[layer],
                self.v[layer],
                self.page_size,
                page_tables,
                row_ids,
                positions,
                keys,
                values,
                write_from,
            ):
                return

        keys = self._cache_payload(keys, dtype=self.k.dtype)
        values = self._cache_payload(values, dtype=self.v.dtype)
        owners = row_ids.long()
        page_index = torch.div(
            positions, self.page_size, rounding_mode="floor"
        ).long()
        pages = page_tables[owners, page_index].long()
        slots = positions.long() - page_index * self.page_size
        flat = pages * self.page_size + slots
        writable = positions >= write_from[owners]
        flat = flat[writable]

        k_flat = self.k[layer].reshape(
            -1, self.num_kv_heads, self.head_dim
        )
        k_flat[flat] = keys[writable]
        if self.v_head_dim:
            v_flat = self.v[layer].reshape(
                -1, self.num_kv_heads, self.v_head_dim
            )
            v_flat[flat] = values[writable]

    def gather_batched(
        self, layer: int, page_tables: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Every page of every row, flattened to tokens: [B, P*page_size, H, D].

        No `seq_len` slicing — that is a host value, and slicing on it would make
        the output shape depend on data a graph cannot re-read. Callers mask the
        tail instead, which is shape-stable across replays.
        """
        index = page_tables.long()
        keys = self.k[layer, index].reshape(
            index.shape[0], -1, self.num_kv_heads, self.head_dim
        )
        if self.v_head_dim:
            values = self.v[layer, index].reshape(
                index.shape[0], -1, self.num_kv_heads, self.v_head_dim
            )
        else:  # MLA: the latent lives in k
            values = keys[:, :, :, :0]
        return keys, values

    def gather(
        self, layer: int, page_table: list[int], seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (K, V) of shape [seq_len, num_kv_heads, head_dim]."""
        num_pages = -(-seq_len // self.page_size)
        index = torch.tensor(page_table[:num_pages], dtype=torch.long, device=self.k.device)
        keys = self.k[layer, index].reshape(-1, self.num_kv_heads, self.head_dim)[:seq_len]
        if self.v_head_dim:
            values = self.v[layer, index].reshape(-1, self.num_kv_heads, self.v_head_dim)[
                :seq_len
            ]
        else:  # MLA: the latent lives in k; v is a width-0 placeholder
            values = keys[:, :, :0]
        return keys, values
