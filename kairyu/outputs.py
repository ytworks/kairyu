"""vLLM-compatible request/completion output types (design doc D2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TokenLogprob:
    """Rich per-token logprob (m9 D3): OpenAI needs token strings + bytes.

    ``bytes_`` is the lossless form — byte-level BPE fragments may decode to
    U+FFFD in ``token``. ``top`` entries carry no nested ``top`` of their own.
    """

    token: str
    token_id: int
    logprob: float
    bytes_: tuple[int, ...] | None = None
    top: tuple[TokenLogprob, ...] = ()


@dataclass(frozen=True)
class CompletionOutput:
    index: int
    text: str
    token_ids: tuple[int, ...]
    cumulative_logprob: float | None = None
    logprobs: tuple[dict[int, float], ...] | None = None
    finish_reason: str | None = None
    stop_reason: int | str | None = None
    # rich server-facing form (m9 D3); id-keyed `logprobs` stays for vLLM compat
    logprob_content: tuple[TokenLogprob, ...] | None = None


@dataclass(frozen=True)
class RequestOutput:
    request_id: str
    prompt: str | None
    prompt_token_ids: tuple[int, ...]
    outputs: tuple[CompletionOutput, ...]
    prompt_logprobs: tuple[dict[int, float], ...] | None = None
    finished: bool = True
    metrics: Any = None
