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
    ) -> None:
        shape = (num_layers, num_pages, page_size, num_kv_heads, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.num_layers = num_layers
        self.num_pages = num_pages
        self.page_size = page_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

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
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=dtype,
            device=device,
        )

    @property
    def bytes_per_token(self) -> int:
        element = self.k.element_size()
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * element

    def _flat_indices(self, page_table: list[int], positions: torch.Tensor) -> torch.Tensor:
        pages = torch.tensor(page_table, dtype=torch.long)[positions // self.page_size]
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
        self.k[layer].reshape(-1, self.num_kv_heads, self.head_dim)[flat] = keys
        self.v[layer].reshape(-1, self.num_kv_heads, self.head_dim)[flat] = values

    def gather(
        self, layer: int, page_table: list[int], seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (K, V) of shape [seq_len, num_kv_heads, head_dim]."""
        num_pages = -(-seq_len // self.page_size)
        index = torch.tensor(page_table[:num_pages], dtype=torch.long)
        keys = self.k[layer, index].reshape(-1, self.num_kv_heads, self.head_dim)[:seq_len]
        values = self.v[layer, index].reshape(-1, self.num_kv_heads, self.head_dim)[:seq_len]
        return keys, values
