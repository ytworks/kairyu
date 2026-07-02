"""vLLM-signature-compatible sampling parameters.

Field names and defaults mirror ``vllm.SamplingParams`` so vLLM examples run
with an import rewrite only (design doc D2).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field


def _normalize_stop(stop: str | Sequence[str] | None) -> tuple[str, ...]:
    if stop is None:
        return ()
    if isinstance(stop, str):
        return (stop,)
    return tuple(stop)


@dataclass(frozen=True)
class SamplingParams:
    n: int = 1
    best_of: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    seed: int | None = None
    stop: str | Sequence[str] | None = None
    stop_token_ids: Sequence[int] | None = None
    max_tokens: int | None = 16
    min_tokens: int = 0
    logprobs: int | None = None
    prompt_logprobs: int | None = None
    ignore_eos: bool = False
    skip_special_tokens: bool = True
    extra_args: dict = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "stop", _normalize_stop(self.stop))
        object.__setattr__(
            self, "stop_token_ids", tuple(self.stop_token_ids) if self.stop_token_ids else ()
        )
        self._validate()

    def _validate(self) -> None:
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError(f"top_k must be -1 or >= 1, got {self.top_k}")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}")
        if self.repetition_penalty <= 0.0:
            raise ValueError(f"repetition_penalty must be > 0, got {self.repetition_penalty}")
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError(f"presence_penalty must be in [-2, 2], got {self.presence_penalty}")
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError(f"frequency_penalty must be in [-2, 2], got {self.frequency_penalty}")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.min_tokens < 0:
            raise ValueError(f"min_tokens must be >= 0, got {self.min_tokens}")

    def clone(self, **overrides: object) -> SamplingParams:
        """Return a new SamplingParams with the given fields replaced."""
        return dataclasses.replace(self, **overrides)  # type: ignore[arg-type]
