"""Engine-side sampling types (design m8 D2) — pure dataclasses, no torch.

Kept separate from ``sampler.py`` so the scheduler/backend import chain stays
free of torch (a dev/GPU dependency); only runners that actually compute
logits import the ``Sampler``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

_MASK64 = (1 << 64) - 1
_SEED_BITS = (1 << 63) - 1


def stable_request_seed(request_id: str) -> int:
    return int.from_bytes(hashlib.sha256(request_id.encode()).digest()[:8], "big") & _SEED_BITS


def mix_seed(base: int, position: int) -> int:
    """splitmix64-style mixer: plain addition collides across adjacent seeds."""
    x = (base ^ ((position + 1) * 0x9E3779B97F4A7C15)) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (x ^ (x >> 31)) & _SEED_BITS


@dataclass(frozen=True)
class EngineSampling:
    """Engine-side sampling parameters; the default is exact greedy."""

    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    min_p: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    seed: int | None = None
    logprobs: int | None = None  # None: no report; N >= 0: chosen + top-N
    json_schema: dict | None = None
    json_mode: bool = False

    @property
    def needs_grammar(self) -> bool:
        return self.json_schema is not None or self.json_mode

    @property
    def is_greedy_pure(self) -> bool:
        """Greedy with no penalties/grammar — the spec-decode-eligible mode."""
        return (
            self.temperature == 0.0
            and self.presence_penalty == 0.0
            and self.frequency_penalty == 0.0
            and self.repetition_penalty == 1.0
            and not self.needs_grammar
        )


@dataclass(frozen=True)
class SampledToken:
    token_id: int
    logprob: float | None = None
    top_logprobs: tuple[tuple[int, float], ...] | None = None
    grammar_terminated: bool = False
