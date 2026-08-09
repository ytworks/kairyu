"""Architecture-native cache contracts for hybrid frontier decoders.

Kairyu owns admission, prefix identity, lifetime, rollback, and reporting.  The
opaque state payload is produced by the architecture implementation (currently
the Transformers model shipped with the pinned Kairyu image); it is never
treated as a generic paged KV tensor.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from kairyu.models.config import ModelConfig


@dataclass(frozen=True)
class CacheComponentDescriptor:
    name: str
    kind: str
    layer_count: int
    dtype: str
    block_size: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.kind:
            raise ValueError("cache component name and kind must be non-empty")
        if self.layer_count < 1:
            raise ValueError("cache component layer_count must be positive")
        if self.block_size is not None and self.block_size < 1:
            raise ValueError("cache component block_size must be positive")


@dataclass(frozen=True)
class CacheDescriptor:
    family: str
    max_context_tokens: int
    components: tuple[CacheComponentDescriptor, ...]
    prefix_policy: str = "complete-state-snapshots"
    speculative_policy: str = "copy-on-write"
    supported_expert_parallel_sizes: tuple[int, ...] = (1,)

    def __post_init__(self) -> None:
        if not self.family:
            raise ValueError("cache family must be non-empty")
        if self.max_context_tokens < 1:
            raise ValueError("cache max_context_tokens must be positive")
        if not self.components:
            raise ValueError("cache descriptor must contain at least one component")

    def as_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "max_context_tokens": self.max_context_tokens,
            "prefix_policy": self.prefix_policy,
            "speculative_policy": self.speculative_policy,
            "supported_expert_parallel_sizes": list(
                self.supported_expert_parallel_sizes
            ),
            "components": [
                {
                    "name": component.name,
                    "kind": component.kind,
                    "layer_count": component.layer_count,
                    "dtype": component.dtype,
                    "block_size": component.block_size,
                    "metadata": component.metadata,
                }
                for component in self.components
            ],
        }


def cache_descriptor_for_model(config: ModelConfig) -> CacheDescriptor:
    frontier = config.frontier_cache
    if frontier is None:
        raise ValueError(
            f"{config.architecture} has no architecture-native cache descriptor"
        )
    counts = {
        layer_type: frontier.layer_types.count(layer_type)
        for layer_type in set(frontier.layer_types)
    }
    if frontier.family == "qwen3.5-hybrid-deltanet":
        return CacheDescriptor(
            family=frontier.family,
            max_context_tokens=config.max_position_embeddings,
            components=(
                CacheComponentDescriptor(
                    name="linear-state",
                    kind="gated-deltanet-recurrent",
                    layer_count=counts["linear_attention"],
                    dtype=frontier.recurrent_state_dtype or "float32",
                    metadata={
                        "conv_kernel": frontier.linear_conv_kernel_dim,
                        "key_heads": frontier.linear_num_key_heads,
                        "value_heads": frontier.linear_num_value_heads,
                        "key_head_dim": frontier.linear_key_head_dim,
                        "value_head_dim": frontier.linear_value_head_dim,
                    },
                ),
                CacheComponentDescriptor(
                    name="full-attention-kv",
                    kind="paged-kv",
                    layer_count=counts["full_attention"],
                    dtype="bfloat16",
                    block_size=frontier.block_size,
                ),
            ),
        )
    if frontier.family == "deepseek-v4-hca-csa":
        return CacheDescriptor(
            family=frontier.family,
            max_context_tokens=config.max_position_embeddings,
            supported_expert_parallel_sizes=(1, 2, 4, 8),
            components=(
                CacheComponentDescriptor(
                    name="hca",
                    kind="heavily-compressed-attention",
                    layer_count=counts["heavily_compressed_attention"],
                    dtype="float8_e4m3fn",
                    block_size=frontier.block_size,
                    metadata={
                        "compress_rate": frontier.compress_rate_hca,
                        "sliding_window": frontier.sliding_window,
                    },
                ),
                CacheComponentDescriptor(
                    name="csa",
                    kind="compressed-sparse-attention",
                    layer_count=counts["compressed_sparse_attention"],
                    dtype="float8_e4m3fn",
                    block_size=frontier.block_size,
                    metadata={
                        "compress_rate": frontier.compress_rate_csa,
                        "sliding_window": frontier.sliding_window,
                        "index_topk": frontier.index_topk,
                        "index_heads": frontier.index_n_heads,
                        "index_head_dim": frontier.index_head_dim,
                        "fp4_indexer_cache": frontier.fp4_indexer_cache,
                    },
                ),
                CacheComponentDescriptor(
                    name="hyper-connection",
                    kind="mhc-state",
                    layer_count=config.num_hidden_layers,
                    dtype="float32",
                    metadata={
                        "streams": frontier.hc_mult,
                        "sinkhorn_iters": frontier.hc_sinkhorn_iters,
                    },
                ),
            ),
        )
    raise ValueError(f"unknown frontier cache family {frontier.family!r}")


def _clone_state(value: Any) -> Any:
    """Clone one opaque cache without retaining mutable tensor aliases."""

    try:
        import torch
    except ImportError:  # pragma: no cover - engine installs torch
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_clone_state(item) for item in value)
    if isinstance(value, list):
        return [_clone_state(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_state(item) for key, item in value.items()}
    # Transformers Cache subclasses contain Python metadata plus tensors.  Its
    # deepcopy implementation recursively clones those tensors; custom cache
    # implementations that cannot be copied fail closed at admission.
    return copy.deepcopy(value)


def _state_nbytes(value: Any, seen: set[int] | None = None) -> int:
    seen = seen if seen is not None else set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    try:
        import torch
    except ImportError:  # pragma: no cover - engine installs torch
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_state_nbytes(item, seen) for item in value.values())
    if isinstance(value, (tuple, list, set)):
        return sum(_state_nbytes(item, seen) for item in value)
    values = getattr(value, "__dict__", None)
    return _state_nbytes(values, seen) if isinstance(values, dict) else 0


@dataclass
class CacheHandle:
    request_id: str
    token_ids: tuple[int, ...] = ()
    state: Any = None
    last_logits: Any = None
    output_epoch: int = 0
    _transaction: tuple[tuple[int, ...], Any, Any, int] | None = None

    def replace(
        self,
        token_ids: tuple[int, ...],
        state: Any,
        last_logits: Any,
        output_epoch: int,
    ) -> None:
        self.token_ids = token_ids
        self.state = state
        self.last_logits = last_logits
        self.output_epoch = output_epoch

    def begin_transaction(self) -> None:
        if self._transaction is not None:
            raise RuntimeError("nested cache transactions are not supported")
        self._transaction = (
            self.token_ids,
            _clone_state(self.state),
            _clone_state(self.last_logits),
            self.output_epoch,
        )

    def commit(self) -> None:
        if self._transaction is None:
            raise RuntimeError("no cache transaction is active")
        self._transaction = None

    def rollback(self) -> None:
        if self._transaction is None:
            raise RuntimeError("no cache transaction is active")
        self.token_ids, self.state, self.last_logits, self.output_epoch = self._transaction
        self._transaction = None


@dataclass(frozen=True)
class _PrefixSnapshot:
    token_ids: tuple[int, ...]
    state: Any
    nbytes: int


class PrefixStateStore:
    """Byte-bounded LRU of complete recurrent/compressed prefix states."""

    def __init__(self, capacity_bytes: int = 0) -> None:
        if type(capacity_bytes) is not int or capacity_bytes < 0:
            raise ValueError("prefix state capacity must be a non-negative integer")
        self.capacity_bytes = capacity_bytes
        self._used_bytes = 0
        self._entries: OrderedDict[tuple[int, ...], _PrefixSnapshot] = OrderedDict()

    @property
    def used_bytes(self) -> int:
        return self._used_bytes

    def put(self, token_ids: tuple[int, ...], state: Any) -> bool:
        if not token_ids or self.capacity_bytes == 0:
            return False
        cloned = _clone_state(state)
        nbytes = _state_nbytes(cloned)
        if nbytes <= 0 or nbytes > self.capacity_bytes:
            return False
        previous = self._entries.pop(token_ids, None)
        if previous is not None:
            self._used_bytes -= previous.nbytes
        snapshot = _PrefixSnapshot(token_ids, cloned, nbytes)
        self._entries[token_ids] = snapshot
        self._used_bytes += nbytes
        while self._used_bytes > self.capacity_bytes:
            _key, evicted = self._entries.popitem(last=False)
            self._used_bytes -= evicted.nbytes
        return True

    def longest_prefix(self, token_ids: tuple[int, ...]) -> tuple[tuple[int, ...], Any] | None:
        matches = [
            key
            for key in self._entries
            if len(key) <= len(token_ids) and token_ids[: len(key)] == key
        ]
        if not matches:
            return None
        key = max(matches, key=len)
        snapshot = self._entries.pop(key)
        self._entries[key] = snapshot
        return key, _clone_state(snapshot.state)

    def clear(self) -> None:
        self._entries.clear()
        self._used_bytes = 0
