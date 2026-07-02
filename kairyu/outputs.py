"""vLLM-compatible request/completion output types (design doc D2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CompletionOutput:
    index: int
    text: str
    token_ids: tuple[int, ...]
    cumulative_logprob: float | None = None
    logprobs: tuple[dict[int, float], ...] | None = None
    finish_reason: str | None = None
    stop_reason: int | str | None = None


@dataclass(frozen=True)
class RequestOutput:
    request_id: str
    prompt: str | None
    prompt_token_ids: tuple[int, ...]
    outputs: tuple[CompletionOutput, ...]
    prompt_logprobs: tuple[dict[int, float], ...] | None = None
    finished: bool = True
    metrics: Any = None
