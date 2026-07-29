"""FlashAttention 3/4 prefill adapters with an explicit FlashInfer decode.

FlashAttention is used only for multi-token prefill.  Decode, including the
tensor-input CUDA-graph seam, stays owned by ``FlashInferBackend``.  Keeping
that split explicit avoids quietly changing Kairyu's already verified
capture-safe decode path when an experimental prefill kernel is selected.

The two upstream generations expose different paged-KV APIs:

* FA3 uses ``flash_attn_with_kvcache`` directly over the physical page pool.
* FA4 beta uses ``flash_attn_varlen_func`` for paged KV.  That path is used
  only on SM100/SM110; SM90 and SM120 materialize the selected pages on-device
  before calling dense ``flash_attn_func``.  In particular, the current SM120
  kernel rejects ``page_table``.

Imports are deferred so CPU installations can still import the engine.  The
module and decode backend injection points also make the exact upstream API
contract CPU-testable without pretending to validate a GPU kernel.
"""

from __future__ import annotations

import importlib
from importlib import metadata
from typing import Any

import torch

from kairyu.engine.core.kv_pool import PagedKVPool

_FA3_SUPPORTED_SMS = frozenset({80, 90})
_FA4_SUPPORTED_SMS = frozenset({90, 100, 110, 120})
_FA4_DIRECT_PAGED_SMS = frozenset({100, 110})


def _normalise_sm(capability: int | tuple[int, int]) -> int:
    if isinstance(capability, tuple):
        if len(capability) != 2:
            raise ValueError("CUDA capability must be an SM integer or a (major, minor) pair")
        major, minor = capability
        return int(major) * 10 + int(minor)
    return int(capability)


def _resolve_sm(
    profile: object | None,
    capability: int | tuple[int, int] | None,
    device: str,
) -> int:
    if capability is not None:
        return _normalise_sm(capability)
    profile_sm = getattr(profile, "sm", None)
    if profile_sm is not None:
        return int(profile_sm)
    if torch.cuda.is_available():
        return _normalise_sm(torch.cuda.get_device_capability(device))
    raise RuntimeError(
        "FlashAttention requires a CUDA hardware profile; pass profile= with "
        "its SM capability (or capability= for an injected CPU contract test)"
    )


def _load_flashattention(generation: int) -> object:
    module_names = (
        ("flash_attn_interface", "flash_attn_3.flash_attn_interface")
        if generation == 3
        else ("flash_attn.cute",)
    )
    required = (
        ("flash_attn_with_kvcache",)
        if generation == 3
        else ("flash_attn_func", "flash_attn_varlen_func")
    )
    errors: list[str] = []
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")
            continue
        missing = [name for name in required if not callable(getattr(module, name, None))]
        if missing:
            errors.append(f"{module_name}: missing {', '.join(missing)}")
            continue
        return module
    if generation == 3:
        action = (
            "install the official FlashAttention-3 package from Dao-AILab/flash-attention/hopper"
        )
    else:
        action = "run `uv sync --extra flashattention4`"
    detail = "; ".join(errors)
    raise RuntimeError(
        f"flashattention{generation} could not import its upstream module; "
        f"{action}. Original errors: {detail}"
    )


def _require_callable(module: object, name: str, generation: int):
    function = getattr(module, name, None)
    if not callable(function):
        version = getattr(module, "__version__", "unknown")
        raise RuntimeError(
            f"flashattention{generation} API mismatch (version={version}): "
            f"expected callable {name!r}"
        )
    return function


def _first_output(result: Any, generation: int) -> torch.Tensor:
    # FA4 returns ``(out, lse)`` even when LSE was not requested.  FA3 returns
    # the output tensor directly unless its optional LSE flag is enabled.
    output = result[0] if isinstance(result, tuple) else result
    if not isinstance(output, torch.Tensor):
        raise RuntimeError(
            f"flashattention{generation} returned {type(output).__name__}, "
            "expected a torch.Tensor output"
        )
    return output


def _module_version(module: object, generation: int) -> str:
    reported = str(getattr(module, "__version__", "unknown"))
    # beta24's ``flash_attn.cute`` currently asks metadata for ``fa4`` while
    # the published distribution is named ``flash-attn-4``, yielding 0.0.0.
    if reported not in ("unknown", "0.0.0"):
        return reported
    distribution = "flash_attn_3" if generation == 3 else "flash-attn-4"
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return reported


class FlashAttentionBackend:
    """Composite prefill/decode backend for one FlashAttention generation."""

    def __init__(
        self,
        generation: int,
        profile: object | None = None,
        *,
        flashattention_module: object | None = None,
        decode_backend: object | None = None,
        device: str = "cuda",
        capability: int | tuple[int, int] | None = None,
    ) -> None:
        if generation not in (3, 4):
            raise ValueError(f"unsupported FlashAttention generation {generation}; expected 3 or 4")
        sm = _resolve_sm(profile, capability, device)
        supported = _FA3_SUPPORTED_SMS if generation == 3 else _FA4_SUPPORTED_SMS
        if sm not in supported:
            supported_text = ", ".join(f"SM{value}" for value in sorted(supported))
            raise RuntimeError(
                f"flashattention{generation} does not support SM{sm}; "
                f"supported profiles: {supported_text}"
            )

        module = (
            flashattention_module
            if flashattention_module is not None
            else _load_flashattention(generation)
        )
        if generation == 3:
            self._fa3_prefill = _require_callable(module, "flash_attn_with_kvcache", generation)
            self._fa4_dense = None
            self._fa4_paged = None
        else:
            self._fa3_prefill = None
            self._fa4_dense = _require_callable(module, "flash_attn_func", generation)
            self._fa4_paged = _require_callable(module, "flash_attn_varlen_func", generation)

        if decode_backend is None:
            from kairyu.engine.core.attention.flashinfer_gpu import (
                FlashInferBackend,
            )

            decode_backend = FlashInferBackend(device=device)
        missing = [
            name
            for name in ("attend", "attend_batched", "plan_decode", "attend_decode")
            if not callable(getattr(decode_backend, name, None))
        ]
        if missing:
            raise TypeError(
                "FlashAttention decode_backend does not satisfy the attention "
                f"contract; missing callables: {', '.join(missing)}"
            )

        kv_mode = (
            "paged-direct"
            if generation == 3 or sm in _FA4_DIRECT_PAGED_SMS
            else "paged-materialized"
        )
        self.generation = generation
        self.sm = sm
        self.profile = profile
        self.prefill_kv_mode = kv_mode
        self.components = {
            "prefill": f"flashattention{generation}",
            "decode": "flashinfer",
            "kv_mode": kv_mode,
        }
        self.component_versions = {
            "prefill": _module_version(module, generation),
            "decode": str(
                getattr(
                    decode_backend,
                    "__version__",
                    getattr(type(decode_backend), "__version__", "unknown"),
                )
            ),
        }
        self._decode_backend = decode_backend
        self.supports_graph_capture = bool(getattr(decode_backend, "supports_graph_capture", False))

    def _supports_head_dim(self, head_dim: int) -> bool:
        if head_dim < 8 or head_dim % 8:
            return False
        if self.generation == 3 or self.sm == 90:
            return head_dim <= 256
        if self.sm in _FA4_DIRECT_PAGED_SMS:
            # FA4 has a dedicated SM100/110 hd256 kernel, but it rejects the
            # ``seqused_k`` input needed for arbitrary partially filled Kairyu
            # pages.  Keep the serving contract fail-closed.
            return head_dim <= 128
        # SM120's 99-KiB shared-memory check permits equal Q/K/V dimensions
        # through 192 with the beta24 tile choices.
        return head_dim <= 192

    def validate_model_config(self, config: object) -> None:
        """Fail before model construction for shapes this adapter cannot serve."""
        if bool(getattr(config, "is_mla", False)):
            raise ValueError(
                f"flashattention{self.generation} MHA adapter does not support MLA models"
            )
        num_qo_heads = int(config.num_attention_heads)
        num_kv_heads = int(config.num_key_value_heads)
        if num_kv_heads <= 0 or num_qo_heads % num_kv_heads:
            raise ValueError(
                "FlashAttention GQA requires num_attention_heads to be "
                "divisible by num_key_value_heads "
                f"({num_qo_heads} vs {num_kv_heads})"
            )
        head_dim = int(config.head_dim)
        if not self._supports_head_dim(head_dim):
            raise ValueError(
                f"flashattention{self.generation} does not support "
                f"head_dim={head_dim} on SM{self.sm} in Kairyu's paged MHA "
                "adapter"
            )
        v_head_dim = int(getattr(config, "kv_cache_v_head_dim", head_dim))
        if v_head_dim != head_dim:
            raise ValueError(
                "Kairyu's FlashAttention MHA adapter requires equal K/V head "
                f"dimensions ({head_dim} vs {v_head_dim})"
            )

    def _validate_prefill(
        self,
        query: torch.Tensor,
        kv_pool: PagedKVPool,
        layer: int,
        page_table: list[int],
        seq_len: int,
        chunk_start: int,
    ) -> int:
        if query.ndim != 3:
            raise ValueError("FlashAttention query must have shape [tokens, heads, head_dim]")
        chunk_len, num_qo_heads, query_head_dim = query.shape
        if chunk_len <= 1:
            raise ValueError("FlashAttention prefill requires at least two tokens")
        if chunk_start < 0 or seq_len <= 0 or chunk_start + chunk_len != seq_len:
            raise ValueError(
                "FlashAttention causal attention is bottom-right aligned: the "
                "query chunk must be the sequence tail "
                f"(chunk_start={chunk_start}, T={chunk_len}, seq_len={seq_len})"
            )
        if not 0 <= layer < kv_pool.num_layers:
            raise ValueError(f"KV layer {layer} is out of range for {kv_pool.num_layers} layers")
        if num_qo_heads % kv_pool.num_kv_heads:
            raise ValueError(
                "FlashAttention GQA requires query heads to be divisible by KV "
                f"heads ({num_qo_heads} vs {kv_pool.num_kv_heads})"
            )
        if query_head_dim != kv_pool.head_dim:
            raise ValueError(
                "FlashAttention query and KV head dimensions differ "
                f"({query_head_dim} vs {kv_pool.head_dim})"
            )
        if not self._supports_head_dim(query_head_dim):
            raise ValueError(
                f"flashattention{self.generation} does not support "
                f"head_dim={query_head_dim} on SM{self.sm} in Kairyu's paged "
                "MHA adapter"
            )
        if kv_pool.v_head_dim != kv_pool.head_dim:
            raise ValueError(
                "Kairyu's FlashAttention MHA adapter requires equal K/V head "
                f"dimensions ({kv_pool.head_dim} vs {kv_pool.v_head_dim})"
            )
        if query.dtype != kv_pool.k.dtype or query.dtype != kv_pool.v.dtype:
            raise ValueError(
                "FlashAttention query/K/V dtypes must match "
                f"({query.dtype}, {kv_pool.k.dtype}, {kv_pool.v.dtype})"
            )
        if query.device != kv_pool.k.device or query.device != kv_pool.v.device:
            raise ValueError(
                "FlashAttention query and KV pool must be on the same device "
                f"({query.device}, {kv_pool.k.device}, {kv_pool.v.device})"
            )

        page_count = -(-seq_len // kv_pool.page_size)
        if len(page_table) < page_count:
            raise ValueError(
                f"page table has {len(page_table)} entries but seq_len={seq_len} "
                f"requires {page_count}"
            )
        selected = page_table[:page_count]
        invalid = [
            page
            for page in selected
            if not isinstance(page, int) or not 0 <= page < kv_pool.num_pages
        ]
        if invalid:
            raise ValueError(f"page table contains invalid physical page ids: {invalid}")
        return page_count

    @staticmethod
    def _page_tensor(
        page_table: list[int],
        page_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.tensor(
            [page_table[:page_count]],
            dtype=torch.int32,
            device=device,
        )

    @staticmethod
    def _cache_length(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.tensor([seq_len], dtype=torch.int32, device=device)

    def _attend_fa3(
        self,
        query: torch.Tensor,
        kv_pool: PagedKVPool,
        layer: int,
        page_table: list[int],
        seq_len: int,
        page_count: int,
    ) -> torch.Tensor:
        result = self._fa3_prefill(  # type: ignore[misc]
            query.unsqueeze(0),
            kv_pool.k[layer],
            kv_pool.v[layer],
            cache_seqlens=self._cache_length(seq_len, query.device),
            page_table=self._page_tensor(page_table, page_count, query.device),
            causal=True,
        )
        return _first_output(result, self.generation)

    @staticmethod
    def _materialize_pages(
        kv_pool: PagedKVPool,
        layer: int,
        page_table: list[int],
        page_count: int,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ``index_select`` copies the physical pages in logical sequence order
        # entirely on the KV pool's device.  No KV payload stages through host.
        index = torch.tensor(
            page_table[:page_count],
            dtype=torch.long,
            device=kv_pool.k.device,
        )
        keys = (
            kv_pool.k[layer]
            .index_select(0, index)
            .flatten(0, 1)[:seq_len]
            .unsqueeze(0)
            .contiguous()
        )
        values = (
            kv_pool.v[layer]
            .index_select(0, index)
            .flatten(0, 1)[:seq_len]
            .unsqueeze(0)
            .contiguous()
        )
        return keys, values

    def _attend_fa4(
        self,
        query: torch.Tensor,
        kv_pool: PagedKVPool,
        layer: int,
        page_table: list[int],
        seq_len: int,
        page_count: int,
    ) -> torch.Tensor:
        query_batched = query.unsqueeze(0)
        if self.sm in _FA4_DIRECT_PAGED_SMS:
            result = self._fa4_paged(  # type: ignore[misc]
                query_batched,
                kv_pool.k[layer],
                kv_pool.v[layer],
                max_seqlen_q=int(query.shape[0]),
                # SM100's paged kernel binds this maximum to page-table width
                # (and its hd256 path requires page-size alignment).  The
                # logical length remains exact in ``seqused_k``.
                max_seqlen_k=page_count * kv_pool.page_size,
                seqused_k=self._cache_length(seq_len, query.device),
                page_table=self._page_tensor(page_table, page_count, query.device),
                causal=True,
            )
        else:
            keys, values = self._materialize_pages(kv_pool, layer, page_table, page_count, seq_len)
            result = self._fa4_dense(  # type: ignore[misc]
                query_batched,
                keys,
                values,
                causal=True,
            )
        return _first_output(result, self.generation)

    def _attend_prefill(
        self,
        query: torch.Tensor,
        kv_pool: PagedKVPool,
        layer: int,
        page_table: list[int],
        seq_len: int,
        chunk_start: int,
    ) -> torch.Tensor:
        page_count = self._validate_prefill(query, kv_pool, layer, page_table, seq_len, chunk_start)
        output = (
            self._attend_fa3(query, kv_pool, layer, page_table, seq_len, page_count)
            if self.generation == 3
            else self._attend_fa4(query, kv_pool, layer, page_table, seq_len, page_count)
        )
        expected = (
            1,
            int(query.shape[0]),
            int(query.shape[1]),
            kv_pool.v_head_dim,
        )
        if tuple(output.shape) != expected:
            raise RuntimeError(
                f"flashattention{self.generation} returned shape "
                f"{tuple(output.shape)}, expected {expected}"
            )
        return output[0].reshape(query.shape[0], -1)

    def attend(
        self,
        query: torch.Tensor,
        kv_pool: PagedKVPool,
        layer: int,
        page_table: list[int],
        seq_len: int,
        chunk_start: int,
    ) -> torch.Tensor:
        if query.ndim == 3 and query.shape[0] == 1:
            return self._decode_backend.attend(
                query, kv_pool, layer, page_table, seq_len, chunk_start
            )
        return self._attend_prefill(query, kv_pool, layer, page_table, seq_len, chunk_start)

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
                f"FlashAttention parallel batch inputs must have the same length ({details})"
            )
        if not queries:
            return []

        outputs: list[torch.Tensor | None] = [None] * len(queries)
        decode_indices = [
            index for index, query in enumerate(queries) if query.ndim == 3 and query.shape[0] == 1
        ]
        if decode_indices:
            decode_outputs = self._decode_backend.attend_batched(
                [queries[index] for index in decode_indices],
                kv_pool,
                layer,
                [page_tables[index] for index in decode_indices],
                [seq_lens[index] for index in decode_indices],
                [chunk_starts[index] for index in decode_indices],
            )
            if len(decode_outputs) != len(decode_indices):
                raise RuntimeError(
                    "FlashInfer decode delegate returned the wrong batch length "
                    f"({len(decode_outputs)} vs {len(decode_indices)})"
                )
            for index, output in zip(decode_indices, decode_outputs, strict=True):
                outputs[index] = output

        decode_set = set(decode_indices)
        for index, query in enumerate(queries):
            if index not in decode_set:
                outputs[index] = self._attend_prefill(
                    query,
                    kv_pool,
                    layer,
                    page_tables[index],
                    seq_lens[index],
                    chunk_starts[index],
                )
        if any(output is None for output in outputs):
            raise RuntimeError("FlashAttention batched dispatch left an output slot empty")
        return [output for output in outputs if output is not None]

    def plan_decode(
        self,
        kv_pool: PagedKVPool,
        page_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        num_qo_heads: int,
        q_dtype: torch.dtype,
    ) -> None:
        return self._decode_backend.plan_decode(
            kv_pool,
            page_tables,
            seq_lens,
            num_qo_heads=num_qo_heads,
            q_dtype=q_dtype,
        )

    def attend_decode(
        self,
        query: torch.Tensor,
        kv_pool: PagedKVPool,
        layer: int,
        page_tables: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        return self._decode_backend.attend_decode(query, kv_pool, layer, page_tables, seq_lens)
