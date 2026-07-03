"""Torch CPU ModelRunner with real paged-KV attention (design m2 §2.5 pre-work).

This is the algorithmic core of the GPU ModelRunner validated with real
tensors on CPU: KV written into non-contiguous pages via the scheduler's page
tables, attention gathered per request, greedy sampling — proven equivalent to
unpaged autoregressive decoding through the full engine (chunked prefill,
compute-skip, radix cache reuse). The GPU phase replaces the naive gather +
matmul attention with FlashInfer kernels behind the same ModelRunner protocol.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from kairyu.engine.core.sampler import Sampler
from kairyu.engine.core.sampling_types import SampledToken
from kairyu.engine.core.scheduler import ScheduledChunk

_DEFAULT_VOCAB = 128
_DEFAULT_DIM = 32


class TinyAttentionLM:
    """Single causal-attention layer with random fixed weights (no MLP/norm).

    Small enough for CPU tests, real enough that any paging bug (wrong page,
    wrong slot, stale KV) changes the sampled tokens.
    """

    def __init__(
        self, vocab: int = _DEFAULT_VOCAB, dim: int = _DEFAULT_DIM, seed: int = 0
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.vocab = vocab
        self.dim = dim
        scale = dim**-0.5
        self.embed = torch.randn(vocab, dim, generator=generator) * scale
        self.wq = torch.randn(dim, dim, generator=generator) * scale
        self.wk = torch.randn(dim, dim, generator=generator) * scale
        self.wv = torch.randn(dim, dim, generator=generator) * scale
        self.wo = torch.randn(dim, dim, generator=generator) * scale
        self.head = torch.randn(dim, vocab, generator=generator) * scale

    def kv_for(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embed[tokens]
        return x @ self.wk, x @ self.wv

    def logits_for_last(
        self, query_token: int, keys: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        """Logits for the query token attending over keys/values (its own KV included)."""
        x = self.embed[query_token]
        q = x @ self.wq
        scores = (keys @ q) * (self.dim**-0.5)
        attention = torch.softmax(scores, dim=0)
        context = attention @ values
        hidden = context @ self.wo + x
        return hidden @ self.head

    def reference_greedy(self, prompt: tuple[int, ...], steps: int) -> tuple[int, ...]:
        """Plain unpaged autoregressive greedy decode (the equivalence oracle)."""
        sequence = list(prompt)
        outputs: list[int] = []
        for _ in range(steps):
            keys, values = self.kv_for(torch.tensor(sequence))
            logits = self.logits_for_last(sequence[-1], keys, values)
            token = int(torch.argmax(logits).item())
            outputs.append(token)
            sequence.append(token)
        return tuple(outputs)


class TorchPagedRunner:
    """ModelRunner writing/reading KV through the scheduler's page tables.

    Without a ``Sampler`` the runner is pure greedy (pre-m8 behavior); with one,
    logits route through the full sampling pipeline (penalties, temperature,
    filters, grammar mask) — same seam the real model runner (M12) uses.
    """

    def __init__(
        self,
        model: TinyAttentionLM,
        num_pages: int,
        page_size: int,
        sampler: Sampler | None = None,
    ) -> None:
        self._model = model
        self._page_size = page_size
        self._sampler = sampler
        self._k_pool = torch.zeros(num_pages, page_size, model.dim)
        self._v_pool = torch.zeros(num_pages, page_size, model.dim)

    def _write_kv(self, token: int, position: int, pages: list[int]) -> None:
        keys, values = self._model.kv_for(torch.tensor([token]))
        page = pages[position // self._page_size]
        slot = position % self._page_size
        self._k_pool[page, slot] = keys[0]
        self._v_pool[page, slot] = values[0]

    def _gather_kv(self, pages: list[int], seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        page_index = torch.tensor(pages[: -(-seq_len // self._page_size)])
        keys = self._k_pool[page_index].reshape(-1, self._model.dim)[:seq_len]
        values = self._v_pool[page_index].reshape(-1, self._model.dim)[:seq_len]
        return keys, values

    def _sample(
        self,
        state: object,
        query_token: int,
        seq_len: int,
        pages: list[int],
        position: int,
    ) -> SampledToken:
        keys, values = self._gather_kv(pages, seq_len)
        logits = self._model.logits_for_last(query_token, keys, values)
        if self._sampler is None:
            return SampledToken(int(torch.argmax(logits).item()))
        return self._sampler.sample(
            state.request.request_id,
            state.request.sampling,
            position,
            logits,
            prompt=state.request.prompt_token_ids,
            outputs=tuple(state.outputs),
            eos_token_id=state.request.eos_token_id,
        )

    def execute(
        self, scheduled: tuple[ScheduledChunk, ...], states: Mapping[str, object]
    ) -> dict[str, tuple[SampledToken, ...]]:
        sampled: dict[str, tuple[SampledToken, ...]] = {}
        for chunk in scheduled:
            state = states[chunk.request_id]
            prompt = state.request.prompt_token_ids
            pages = list(state.allocation.pages) + list(state.decode_pages)
            if chunk.is_prefill:
                end = state.computed_prompt
                for position in range(end - chunk.num_tokens, end):
                    self._write_kv(prompt[position], position, pages)
                if state.prefill_done and end == len(prompt):
                    sampled[chunk.request_id] = (
                        self._sample(
                            state, prompt[-1], seq_len=len(prompt), pages=pages, position=0
                        ),
                    )
            else:
                # decode for output index p: previous token's KV lands at
                # absolute position prompt_len + p - 1, then it queries
                p = chunk.position
                input_token = state.outputs[p - 1] if p > 0 else prompt[-1]
                absolute = len(prompt) + p - 1
                self._write_kv(input_token, absolute, pages)
                sampled[chunk.request_id] = (
                    self._sample(
                        state, input_token, seq_len=absolute + 1, pages=pages, position=p
                    ),
                )
        return sampled
