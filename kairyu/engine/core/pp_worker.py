"""Pipeline-parallel stage execution over dist send/recv (m16 D4/A9).

``PpStageModel`` is the stage seam the m6 ``StageWorker`` protocol needed:
stage 0 embeds, middle stages take hidden tensors, the final stage applies the
norm + lm_head. Each stage's pool holds only ITS layers — layer indices are
rebased by enumerating the stage's own slice. Hidden states travel as
``tensor_send/recv`` between adjacent ranks; the final stage samples.
"""

from __future__ import annotations

import torch
from torch import nn

from kairyu.models.llama import DenseDecoder


def stage_layer_bounds(num_layers: int, num_stages: int, stage: int) -> tuple[int, int]:
    """Contiguous layer slices; remainder layers go to the EARLY stages so the
    final (sampling) stage is never the largest."""
    base = num_layers // num_stages
    extra = num_layers % num_stages
    start = stage * base + min(stage, extra)
    size = base + (1 if stage < extra else 0)
    return start, start + size


class PpStageModel(nn.Module):
    """One pipeline stage cut out of a full DenseDecoder (m16 A9)."""

    def __init__(self, full: DenseDecoder, num_stages: int, stage: int) -> None:
        super().__init__()
        self.config = full.config
        self.stage = stage
        self.num_stages = num_stages
        first, last = stage_layer_bounds(full.config.num_hidden_layers, num_stages, stage)
        self.first_layer = first
        self.num_local_layers = last - first
        self.is_first = stage == 0
        self.is_last = stage == num_stages - 1
        self.layers = nn.ModuleList(full.model.layers[first:last])
        self.rotary_emb = full.model.rotary_emb
        self.embed_tokens = full.model.embed_tokens if self.is_first else None
        self.norm = full.model.norm if self.is_last else None
        self.lm_head = full.lm_head if self.is_last else None

    @torch.no_grad()
    def forward_stage(
        self,
        inputs: torch.Tensor,
        positions: torch.Tensor,
        kv_pool,
        page_table: list[int],
        seq_len: int,
        write_from: int = 0,
    ) -> torch.Tensor:
        """inputs: token ids [T] on stage 0, hidden [T, H] elsewhere.

        Returns post-norm hidden on the last stage (feed ``logits()``),
        raw hidden to send downstream otherwise. Pool layer indices are
        stage-local (0..num_local_layers-1).
        """
        if self.is_first:
            hidden = self.embed_tokens(inputs)
        else:
            hidden = inputs
        cos, sin = self.rotary_emb(positions)
        for local_index, layer in enumerate(self.layers):
            hidden = layer(
                hidden, cos, sin, kv_pool, local_index, page_table,
                positions, seq_len, write_from,
            )
        if self.is_last:
            hidden = self.norm(hidden)
        return hidden

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        if not self.is_last:  # pragma: no cover - guarded by callers
            raise RuntimeError("logits only exist on the final pipeline stage")
        return self.lm_head(hidden)


def pp_greedy_generate(
    stage_model: PpStageModel,
    comm,
    kv_pool,
    page_table: list[int],
    prompt: list[int],
    max_new_tokens: int,
) -> list[int]:
    """Reference PP loop: prefill + decode with hidden handoff between stages.

    Every stage runs this same function (SPMD): stage 0 feeds tokens, middle/
    final stages receive hidden; the FINAL stage samples greedily and
    broadcasts the token so stage 0 can feed the next step.
    """
    hidden_size = stage_model.config.hidden_size
    tokens = list(prompt)
    outputs: list[int] = []
    computed = 0
    for _ in range(max_new_tokens):
        chunk = tokens[computed:]
        chunk_len = len(chunk)
        positions = torch.arange(computed, computed + chunk_len)
        seq_len = computed + chunk_len
        if stage_model.is_first:
            inputs = torch.tensor(chunk)
        else:
            inputs = torch.empty(chunk_len, hidden_size)
            comm.tensor_recv(inputs, src=stage_model.stage - 1)
        hidden = stage_model.forward_stage(
            inputs, positions, kv_pool, page_table, seq_len, write_from=computed
        )
        if not stage_model.is_last:
            comm.tensor_send(hidden.contiguous(), dst=stage_model.stage + 1)
        if stage_model.is_last:
            token = int(torch.argmax(stage_model.logits(hidden[-1])).item())
        else:
            token = 0
        token = comm.broadcast(token, src=stage_model.num_stages - 1)
        computed = seq_len
        outputs.append(token)
        tokens.append(token)
    return outputs
