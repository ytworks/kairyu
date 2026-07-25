"""AttentionBackend seam (m13 D1).

Backends are plain objects satisfying the protocol — NEVER nn.Module (a
stateful backend like FlashInfer's workspace must not register as a submodule
or checkpoints grow bogus keys). ``query`` positions are contiguous from
``chunk_start`` (documented invariant all call sites satisfy).
"""

from typing import Protocol

import torch

from kairyu.engine.core.kv_pool import PagedKVPool


class AttentionBackend(Protocol):
    def attend(
        self,
        query: torch.Tensor,
        kv_pool: PagedKVPool,
        layer: int,
        page_table: list[int],
        seq_len: int,
        chunk_start: int,
    ) -> torch.Tensor:
        """query [T, heads, head_dim] -> context [T, heads * head_dim]."""
        ...

    def attend_batched(
        self,
        queries: list[torch.Tensor],
        kv_pool: PagedKVPool,
        layer: int,
        page_tables: list[list[int]],
        seq_lens: list[int],
        chunk_starts: list[int],
    ) -> list[torch.Tensor]:
        """Per-sequence contexts, identical to per-sequence ``attend`` (C4).

        The batched seam the GPU runner needs: one call per layer per step over N
        sequences instead of N calls. Backends may batch the kernel internally."""
        ...


class GraphDecodeBackend(Protocol):
    """The CUDA-graph decode capability contract (m17 D1, review [P1]).

    THE canonical definition — #138 (the runner's fail-fast gate) and #141
    (FlashInfer) both refer to this one, nobody restates it.

    Owning ``attend_decode`` is NOT what makes a backend capturable. Every
    kernel a captured region runs must be free of host synchronization:
    ``.tolist()``, ``.cpu()``, ``.item()``, or anything that reads a device
    value, raises inside ``torch.cuda.graph``. FlashInfer's ``plan()`` is
    exactly that — it copies ``indptr`` to the host to build the split-KV
    schedule, and its own docs say it "cannot be used in Cuda Graph". So a
    backend can implement ``attend_decode`` perfectly and still make the first
    capture die.

    The contract is therefore three members, and a backend must honor all
    three before ``PagedModelRunner`` will capture it:

    ``supports_graph_capture``
        Declared, not inferred. True asserts that ``attend_decode`` performs
        NO host synchronization once ``plan_decode`` has run for the step —
        which in practice means the backend keeps its planning state in
        static/pinned device buffers a captured kernel can keep pointing at.
        Absent or False means the runner refuses to build a graph path.

    ``plan_decode``
        The STEP-BOUNDARY host hook. Everything that must touch the host goes
        here, and it runs outside the captured region: once before capture,
        and once after each ``_copy_in`` before ``replay()`` — the arguments
        are the executor's static PADDED buffers, whose contents the replay is
        about to attend over. A backend with no host phase implements it as a
        no-op (see ``TorchAttentionBackend``) so the hook is always callable
        and the executor never has to sniff for it per step.

    ``attend_decode``
        The captured half: tensors in, tensors out, no host values.
    """

    supports_graph_capture: bool

    def plan_decode(
        self,
        kv_pool: PagedKVPool,
        page_tables: torch.Tensor,  # [B, P] int
        seq_lens: torch.Tensor,  # [B] int
        *,
        num_qo_heads: int,
        q_dtype: torch.dtype,
    ) -> None:
        """Host phase for ONE decode step, over the buffers about to be read."""
        ...

    def attend_decode(
        self,
        query: torch.Tensor,  # [B, heads, head_dim]
        kv_pool: PagedKVPool,
        layer: int,
        page_tables: torch.Tensor,  # [B, P] int
        seq_lens: torch.Tensor,  # [B] int
    ) -> torch.Tensor:
        """Capture-safe decode attention -> [B, heads * head_dim]."""
        ...


def graph_capture_gap(backend: object) -> str | None:
    """Why ``backend`` cannot be captured, as a phrase, or None if it can.

    The single enforcement point for ``GraphDecodeBackend``. Checked at runner
    construction so an unsupported backend fails where an operator can act on
    it, instead of at the first capture (a ``RuntimeError`` from a D2H copy) or
    the first batched decode (an ``AttributeError``) arbitrarily far into a run.
    """
    name = type(backend).__name__
    if not callable(getattr(backend, "attend_decode", None)):
        return f"{name} has no attend_decode"
    if not getattr(backend, "supports_graph_capture", False):
        return (
            f"{name} has attend_decode but does not declare "
            "supports_graph_capture — a backend that synchronizes with the host "
            "inside the captured region (FlashInfer's plan(), .tolist(), .cpu()) "
            "must not be captured"
        )
    if not callable(getattr(backend, "plan_decode", None)):
        return (
            f"{name} declares supports_graph_capture but has no plan_decode "
            "step-boundary hook (use a no-op if there is no host phase)"
        )
    return None


from kairyu.engine.core.attention.selector import (  # noqa: E402
    select_backend,
    select_backend_name,
)
from kairyu.engine.core.attention.torch_backend import TorchAttentionBackend  # noqa: E402

__all__ = [
    "AttentionBackend",
    "GraphDecodeBackend",
    "TorchAttentionBackend",
    "graph_capture_gap",
    "select_backend",
    "select_backend_name",
]
