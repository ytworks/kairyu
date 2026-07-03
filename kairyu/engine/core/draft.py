"""DraftSource: pluggable draft-token proposers for speculative decode (m17 D3).

``NGramDraftSource`` wraps the m8 ``propose_ngram`` (default — behavior
byte-identical). ``ModelDraftSource`` rolls a draft head autoregressively.
The verify path is draft-agnostic (m8 ``verify_greedy`` unchanged), and every
scheduler invariant (k+1 reservation, spec_in_flight shortfall release,
degrade-to-1) lives ABOVE this seam.
"""

from __future__ import annotations

from typing import Protocol

import torch

from kairyu.engine.core.spec_decode import propose_ngram


class DraftSource(Protocol):
    def propose(self, context: tuple[int, ...], max_draft: int) -> list[int]:
        """Draft continuation of ``context`` (prompt + committed outputs)."""
        ...


class NGramDraftSource:
    """m8 n-gram matcher behind the protocol (the free-draft baseline)."""

    def propose(self, context: tuple[int, ...], max_draft: int) -> list[int]:
        return propose_ngram(context, max_draft=max_draft)


class ModelDraftSource:
    """Autoregressive draft head (EAGLE/MTP class): greedy k-token rollout.

    ``draft_model`` contract: ``draft_next(token_ids) -> next_token_logits``
    over the running context — the head classes provide it (m17 D4/D5).
    """

    def __init__(self, draft_model) -> None:
        self._draft_model = draft_model

    def propose(self, context: tuple[int, ...], max_draft: int) -> list[int]:
        if max_draft <= 0:
            return []
        tokens = list(context)
        drafted: list[int] = []
        for _ in range(max_draft):
            logits = self._draft_model.draft_next(tuple(tokens))
            token = int(torch.argmax(logits).item())
            drafted.append(token)
            tokens.append(token)
        return drafted
