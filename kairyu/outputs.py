"""vLLM-compatible request/completion output types (design doc D2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class _TextSnapshot:
    """Append-only text parts captured at one streaming boundary."""

    __slots__ = ("_part_count", "_parts", "_value")

    def __init__(self, parts: list[str]) -> None:
        self._parts = parts
        self._part_count = len(parts)
        self._value: str | None = None

    def materialize(self) -> str:
        value = self._value
        if value is None:
            value = "".join(self._parts[: self._part_count])
            self._value = value
        return value


def _lazy_text(parts: list[str]) -> str:
    """Return an internal lazy value while keeping the public type ``str``."""

    return _TextSnapshot(parts)  # type: ignore[return-value]


class _TextField:
    """Materialize only reads of the public ``CompletionOutput.text`` field."""

    def __get__(self, instance: object | None, owner: type | None = None) -> str:
        if instance is None:
            # Descriptor-backed dataclass fields signal a required argument by
            # raising here instead of returning a class-level default.
            raise AttributeError
        value = instance.__dict__["text"]
        return value.materialize() if isinstance(value, _TextSnapshot) else value

    def __set__(self, instance: object, value: str) -> None:
        instance.__dict__["text"] = value


@dataclass(frozen=True)
class TokenLogprob:
    """Rich per-token logprob (m9 D3): OpenAI needs token strings + bytes.

    Native Kairyu entries use the tokenizer's raw vocabulary piece, so byte
    fragments and special tokens remain identifiable; remote adapters preserve
    the upstream token representation. ``bytes_`` is backend-defined decoded
    byte metadata. ``top`` entries carry no nested ``top``.
    """

    token: str
    token_id: int
    logprob: float
    bytes_: tuple[int, ...] | None = None
    top: tuple[TokenLogprob, ...] = ()


@dataclass(frozen=True)
class CompletionOutput:
    index: int
    text: str = _TextField()  # type: ignore[assignment]
    token_ids: tuple[int, ...]
    cumulative_logprob: float | None = None
    logprobs: tuple[dict[int, float], ...] | None = None
    finish_reason: str | None = None
    stop_reason: int | str | None = None
    # rich server-facing form (m9 D3); id-keyed `logprobs` stays for vLLM compat
    logprob_content: tuple[TokenLogprob, ...] | None = None
    # Streaming backends may provide the newly visible suffix directly.  The
    # cumulative ``text`` field remains the compatibility surface for callers
    # that do not consume deltas.
    text_delta: str | None = None
    text_offset: int | None = None

    def delta_after(self, offset: int) -> tuple[str, int]:
        """Return the next stream delta and its cumulative end offset.

        Older backends expose only cumulative ``text`` and retain the prior
        slicing behavior. Delta-native backends avoid reading or scanning that
        cumulative value on the streaming hot path.
        """

        if type(offset) is not int or offset < 0:
            raise ValueError(f"completion text offset must be non-negative, got {offset!r}")
        if self.text_delta is None and self.text_offset is None:
            return self.text[offset:], len(self.text)
        if (
            not isinstance(self.text_delta, str)
            or type(self.text_offset) is not int
            or self.text_offset != offset
        ):
            raise ValueError(
                "completion text delta offset mismatch: "
                f"expected {offset}, got {self.text_offset!r}"
            )
        return self.text_delta, offset + len(self.text_delta)


@dataclass(frozen=True)
class RequestOutput:
    request_id: str
    prompt: str | None
    prompt_token_ids: tuple[int, ...]
    outputs: tuple[CompletionOutput, ...]
    prompt_logprobs: tuple[dict[int, float], ...] | None = None
    finished: bool = True
    metrics: Any = None
