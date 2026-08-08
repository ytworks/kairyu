"""DraftSource: pluggable draft-token proposers for speculative decode (m17 D3).

``NGramDraftSource`` wraps the m8 ``propose_ngram`` (default — behavior
byte-identical). ``ModelDraftSource`` rolls a draft head autoregressively.
The verify path is draft-agnostic (m8 ``verify_greedy`` unchanged), and every
scheduler invariant (k+1 reservation, spec_in_flight shortfall release,
degrade-to-1) lives ABOVE this seam.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import torch

from kairyu.engine.core.spec_decode import propose_ngram


class DraftSource(Protocol):
    def propose(self, context: tuple[int, ...], max_draft: int) -> list[int]:
        """Draft continuation of ``context`` (prompt + committed outputs)."""
        ...


class NGramDraftSource:
    """m8 n-gram matcher behind the protocol (the free-draft baseline)."""

    source_name = "ngram"

    def propose(self, context: tuple[int, ...], max_draft: int) -> list[int]:
        return propose_ngram(context, max_draft=max_draft)


class ModelDraftSource:
    """Autoregressive draft head (EAGLE/MTP class): greedy k-token rollout.

    ``draft_model`` contract: ``draft_next(token_ids) -> next_token_logits``
    over the running context — the head classes provide it (m17 D4/D5).
    """

    def __init__(self, draft_model) -> None:
        self._draft_model = draft_model

    source_name = "model"

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


@dataclass
class _TargetTrace:
    history: torch.Tensor | None = None
    staged: list[torch.Tensor] = field(default_factory=list)
    rows: int = 0
    unavailable: bool = False


class _LearnedDraftAdapter:
    """Target-hidden history shared by a learned proposer and target runner.

    The target runner stages every speculative verification row.  Verification
    then commits only the rows whose input prefix became authoritative: the
    accepted draft prefix plus the target correction/bonus input row.  A RadixKV
    hit can begin after position zero without a corresponding hidden-state
    cache; that request degrades to ordinary target decode instead of using a
    partial, silently-wrong draft context.
    """

    source_name: str
    capture_kind: str

    def __init__(self) -> None:
        self._traces: dict[str, _TargetTrace] = {}

    @staticmethod
    def _append_history(trace: _TargetTrace, rows: torch.Tensor) -> None:
        required = trace.rows + rows.shape[0]
        history = trace.history
        if history is None or history.shape[0] < required:
            capacity = max(16, 1 << (required - 1).bit_length())
            grown = rows.new_empty((capacity, rows.shape[1]))
            if history is not None:
                grown[: trace.rows].copy_(history[: trace.rows])
            trace.history = grown
        trace.history[trace.rows : required].copy_(rows)
        trace.rows = required

    def _transform(
        self,
        hidden: torch.Tensor,
        aux_hidden: torch.Tensor | None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def capture_target_rows(
        self,
        request_id: str,
        positions: tuple[int, ...],
        hidden: torch.Tensor,
        *,
        aux_hidden: torch.Tensor | None = None,
        staged: bool = False,
    ) -> None:
        if hidden.ndim != 2 or hidden.shape[0] != len(positions):
            raise ValueError(
                "draft target capture rows must match their absolute positions"
            )
        trace = self._traces.setdefault(request_id, _TargetTrace())
        if trace.unavailable:
            return
        staged_rows = sum(chunk.shape[0] for chunk in trace.staged)
        expected_start = trace.rows + (staged_rows if staged else 0)
        expected = tuple(range(expected_start, expected_start + len(positions)))
        if positions != expected or (not staged and trace.staged):
            trace.history = None
            trace.staged.clear()
            trace.rows = 0
            trace.unavailable = True
            return
        captured = self._transform(hidden, aux_hidden).detach()
        if captured.ndim != 2 or captured.shape[0] != len(positions):
            raise RuntimeError("learned draft adapter changed target capture row count")
        if staged:
            trace.staged.append(captured)
        else:
            self._append_history(trace, captured)

    def commit_verification(
        self,
        request_id: str,
        accepted_draft_tokens: int,
    ) -> None:
        trace = self._traces.get(request_id)
        if trace is None or trace.unavailable or not trace.staged:
            return
        keep = accepted_draft_tokens + 1
        available = sum(chunk.shape[0] for chunk in trace.staged)
        if not 0 < keep <= available:
            raise RuntimeError(
                "learned draft verification commit exceeds staged target rows"
            )
        remaining = keep
        for chunk in trace.staged:
            if remaining == 0:
                break
            take = min(remaining, chunk.shape[0])
            self._append_history(trace, chunk[:take])
            remaining -= take
        trace.staged.clear()

    def release(self, request_id: str) -> None:
        self._traces.pop(request_id, None)

    def _history(
        self,
        request_id: str,
        context: tuple[int, ...],
    ) -> torch.Tensor | None:
        trace = self._traces.get(request_id)
        expected = len(context) - 1
        if (
            expected < 1
            or trace is None
            or trace.unavailable
            or trace.staged
            or trace.rows != expected
        ):
            return None
        assert trace.history is not None
        return trace.history[: trace.rows]


class EagleDraftAdapter(_LearnedDraftAdapter):
    """EAGLE-3 serving adapter with fused target history and rollout K/V."""

    source_name = "eagle"
    capture_kind = "eagle_aux"

    def __init__(self, head, target_model) -> None:
        super().__init__()
        self._head = head
        self._embedding = target_model.model.embed_tokens

    def _transform(
        self,
        hidden: torch.Tensor,
        aux_hidden: torch.Tensor | None,
    ) -> torch.Tensor:
        del hidden
        if aux_hidden is None:
            raise ValueError("EAGLE target capture requires auxiliary hidden rows")
        return self._head.fuse(aux_hidden)

    @torch.no_grad()
    def propose_for_request(
        self,
        request_id: str,
        context: tuple[int, ...],
        max_draft: int,
    ) -> list[int]:
        if max_draft <= 0:
            return []
        fused = self._history(request_id, context)
        if fused is None:
            return []
        token_ids = torch.tensor(
            context[1:], dtype=torch.long, device=fused.device
        )
        shifted = self._embedding(token_ids)
        return self._head.rollout_cached(
            shifted,
            fused,
            lambda token_id: self._embedding.weight[token_id],
            max_draft,
        )


class MtpDraftAdapter(_LearnedDraftAdapter):
    """DeepSeek MTP serving adapter with one persistent rollout KV pool."""

    source_name = "mtp"
    capture_kind = "final_hidden"

    def __init__(self, head, target_model) -> None:
        super().__init__()
        self._head = head
        self._rotary_emb = target_model.model.rotary_emb

    def _transform(
        self,
        hidden: torch.Tensor,
        aux_hidden: torch.Tensor | None,
    ) -> torch.Tensor:
        del aux_hidden
        return hidden

    @torch.no_grad()
    def propose_for_request(
        self,
        request_id: str,
        context: tuple[int, ...],
        max_draft: int,
    ) -> list[int]:
        if max_draft <= 0:
            return []
        hidden = self._history(request_id, context)
        if hidden is None:
            return []
        token_ids = torch.tensor(
            context[1:], dtype=torch.long, device=hidden.device
        )
        return self._head.rollout_cached(
            token_ids,
            hidden,
            self._rotary_emb,
            max_draft,
        )


def build_learned_draft_source(
    mode: str,
    draft_model_path: str | Path,
    *,
    target_model,
    target_config,
) -> EagleDraftAdapter | MtpDraftAdapter:
    """Load and bind one learned draft head to a resident target model."""

    device = next(target_model.parameters()).device
    dtype = next(target_model.parameters()).dtype
    if mode == "eagle":
        import json

        from kairyu.models.eagle import EagleConfig, load_eagle_head

        path = Path(draft_model_path)
        config_path = path / "config.json"
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except OSError as error:
            raise ValueError(
                f"cannot read EAGLE config.json at {config_path}"
            ) from error
        config = EagleConfig.from_checkpoint_config(raw)
        if config.hidden_size != target_config.hidden_size:
            raise ValueError(
                "EAGLE and target hidden sizes differ: "
                f"{config.hidden_size} != {target_config.hidden_size}"
            )
        head = load_eagle_head(
            path,
            config,
            dtype=dtype,
            target_device=device,
        ).to(device)
        mapped = torch.arange(
            config.draft_vocab_size, dtype=torch.int64, device=device
        ) + head.d2t
        if int(mapped.min().item()) < 0 or int(mapped.max().item()) >= target_config.vocab_size:
            raise ValueError("EAGLE d2t map falls outside the target vocabulary")
        return EagleDraftAdapter(head, target_model)
    if mode == "mtp":
        from kairyu.models.mtp import load_mtp_head

        head = load_mtp_head(
            draft_model_path,
            target_config,
            dtype=dtype,
            target_device=device,
        ).to(device)
        return MtpDraftAdapter(head, target_model)
    raise ValueError(f"unknown learned speculative mode {mode!r}")
