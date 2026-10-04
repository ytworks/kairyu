"""Immutable turn scripts replayed by the scenario test engines (M20 WP-02b).

A ``Scenario`` chooses one ``Turn`` per generation call; a ``Turn`` is an
ordered tuple of parts (visible text, ``Reasoning``, ``ToolCall``) plus the
stream shape (chunking, latency, finish reason, usage, injected failure).

Each engine renders the parts into its own token pieces: ``ScenarioBackend``
emits the generic ``<tool_call>`` envelope and inline or typed reasoning,
while ``FakeVLLMUpstream`` emits OpenAI chat deltas. Both share the toy
tokenizer, the ``max_tokens`` cut and the chunking rule defined here.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

from kairyu.engine.backend import GenerationUsage

FinishReason = Literal["stop", "length", "content_filter"]
_FINISH_REASONS = frozenset({"stop", "length", "content_filter"})

# Whitespace attaches to the following word, like a BPE "Ġword" piece, so the
# pieces of a string always concatenate back to exactly that string.
_TOKEN_PATTERN = re.compile(r"\s*\S+|\s+")
# ``chat_template.render_chat`` (legacy renderer): "role: text" lines, then
# the bare "assistant:" generation prompt (no space, so never a message line).
_LEGACY_USER_MARKER = re.compile(r"(?m)^user: ")
_LEGACY_NEXT_ROLE = re.compile(r"\n(?:assistant|developer|system|tool|user):")
_LEGACY_LATER_MESSAGE = re.compile(r"\n(?:assistant|developer|system|tool): ")
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def toy_tokens(text: str) -> tuple[str, ...]:
    """Split ``text`` into deterministic token pieces (one word each)."""

    return tuple(_TOKEN_PATTERN.findall(text))


def numbered_text(count: int, *, word: str = "w") -> str:
    """Visible text that is exactly ``count`` toy tokens long."""

    if type(count) is not int or count < 0:
        raise ValueError("count must be a non-negative integer")
    if not word or any(char.isspace() for char in word):
        raise ValueError("word must be a non-empty string without whitespace")
    return " ".join(f"{word}{index}" for index in range(count))


def pending_user_text(rendered: str) -> str:
    """The user message a legacy-rendered prompt awaits a reply to.

    Empty once another message (an assistant tool call, a tool result)
    follows the last user message. Model chat templates have no portable
    role markers, so their prompts are returned whole.
    """

    starts = tuple(_LEGACY_USER_MARKER.finditer(rendered))
    if not starts:
        return rendered
    tail = rendered[starts[-1].end() :]
    if _LEGACY_LATER_MESSAGE.search(tail):
        return ""
    end = _LEGACY_NEXT_ROLE.search(tail)
    return tail if end is None else tail[: end.start()]


@dataclass(frozen=True)
class Reasoning:
    """Private reasoning: inline ``<think>`` tags, or the typed channel.

    ``typed=True`` is what a vLLM reasoning parser produces
    (``reasoning``/``reasoning_content`` deltas); inline tags are what a raw
    model emits into visible text.
    """

    text: str
    typed: bool = False


@dataclass(frozen=True)
class ToolCall:
    """One function call; ``arguments`` is frozen as canonical JSON text."""

    name: str
    arguments: str = "{}"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("tool call name must be a non-empty string")
        parsed = json.loads(self.arguments)
        if not isinstance(parsed, dict):
            raise ValueError("tool call arguments must encode a JSON object")
        canonical = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        object.__setattr__(self, "arguments", canonical)

    @classmethod
    def of(cls, name: str, /, **arguments: object) -> ToolCall:
        return cls(name, json.dumps(arguments, ensure_ascii=False))

    def argument_object(self) -> dict[str, object]:
        return json.loads(self.arguments)

    def generic_envelope(self) -> str:
        """The GENERIC-protocol text ``chat_service._parse_tool_calls`` parses."""

        payload = {"name": self.name, "arguments": self.argument_object()}
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return f"<tool_call>{body}</tool_call>"


Part: TypeAlias = str | Reasoning | ToolCall


def inline_part_texts(part: Part) -> tuple[str, ...]:
    """The token pieces of ``part`` as raw model output in the visible text.

    A ``ToolCall`` is its generic envelope and ``Reasoning`` sits inside
    inline think tags. Engines route typed reasoning to their own channel
    before calling this.
    """

    if isinstance(part, ToolCall):
        return toy_tokens(part.generic_envelope())
    if isinstance(part, Reasoning):
        return (THINK_OPEN, *toy_tokens(part.text), THINK_CLOSE)
    return toy_tokens(part)


@dataclass(frozen=True)
class Turn:
    """One scripted generation.

    ``chunk_tokens`` groups that many token pieces per streamed chunk (``None``
    = one chunk per part); a chunk never spans two parts. ``usage`` replaces
    the computed usage verbatim (exact prompt/completion/cached counts).
    ``fail_after_chunks=N`` streams N chunks and then fails the generation.
    A ``max_tokens`` budget below the scripted length cuts the output and
    reports ``finish_reason="length"`` regardless of ``finish_reason``.
    """

    parts: tuple[Part, ...] | Part = ()
    finish_reason: FinishReason = "stop"
    usage: GenerationUsage | None = None
    chunk_tokens: int | None = 1
    first_token_delay_s: float = 0.0
    inter_chunk_delay_s: float = 0.0
    fail_after_chunks: int | None = None

    def __post_init__(self) -> None:
        parts = (
            (self.parts,)
            if isinstance(self.parts, (str, Reasoning, ToolCall))
            else tuple(self.parts)
        )
        if not all(isinstance(part, (str, Reasoning, ToolCall)) for part in parts):
            raise TypeError("turn parts must be str, Reasoning or ToolCall")
        object.__setattr__(self, "parts", parts)
        if self.finish_reason not in _FINISH_REASONS:
            raise ValueError(f"finish_reason must be one of {sorted(_FINISH_REASONS)}")
        if self.chunk_tokens is not None and (
            type(self.chunk_tokens) is not int or self.chunk_tokens < 1
        ):
            raise ValueError("chunk_tokens must be a positive integer or None")
        if self.first_token_delay_s < 0 or self.inter_chunk_delay_s < 0:
            raise ValueError("turn delays must be non-negative")
        if self.fail_after_chunks is not None and (
            type(self.fail_after_chunks) is not int or self.fail_after_chunks < 0
        ):
            raise ValueError("fail_after_chunks must be a non-negative integer or None")

    @property
    def has_typed_reasoning(self) -> bool:
        return any(isinstance(part, Reasoning) and part.typed for part in self.parts)

    @property
    def has_tool_calls(self) -> bool:
        return any(isinstance(part, ToolCall) for part in self.parts)


class UnscriptedCallError(RuntimeError):
    """A generation call matched no rule, turn or default of its scenario."""


@dataclass(frozen=True)
class Scenario:
    """Selects a turn for each generation call.

    Precedence: the first ``rules`` entry whose key occurs in the call's
    pending user text; else ``turns[call_index]`` (0-based dispatch order on
    one engine instance); else ``default``. An unscripted call fails loudly.

    The pending user text is empty once another message (an assistant tool
    call, a tool result) follows the last user message, so a rule answers a
    user message once and the follow-up requests of a tool loop fall through:
    script tool loops with index-based ``turns`` or ``default``. Prompts
    without legacy role markers are matched whole, so there a rule matches
    every request carrying its key.
    """

    turns: tuple[Turn, ...] = ()
    rules: tuple[tuple[str, Turn], ...] | Mapping[str, Turn] = ()
    default: Turn | None = None

    def __post_init__(self) -> None:
        rules = (
            tuple(self.rules.items())
            if isinstance(self.rules, Mapping)
            else tuple(tuple(rule) for rule in self.rules)
        )
        for rule in rules:
            if len(rule) != 2 or not isinstance(rule[0], str) or not rule[0]:
                raise ValueError("scenario rules map non-empty text keys to turns")
            if not isinstance(rule[1], Turn):
                raise TypeError("scenario rules map text keys to Turn values")
        turns = tuple(self.turns)
        if not all(isinstance(turn, Turn) for turn in turns):
            raise TypeError("scenario turns must be Turn values")
        if self.default is not None and not isinstance(self.default, Turn):
            raise TypeError("scenario default must be a Turn or None")
        object.__setattr__(self, "rules", rules)
        object.__setattr__(self, "turns", turns)

    def select(self, call_index: int, user_text: str) -> Turn:
        for key, turn in self.rules:
            if key in user_text:
                return turn
        if call_index < len(self.turns):
            return self.turns[call_index]
        if self.default is not None:
            return self.default
        raise UnscriptedCallError(
            f"scenario has no turn for call {call_index} "
            f"(pending user text {user_text[-80:]!r})"
        )


class ScriptedFailure(RuntimeError):
    """The failure a turn's ``fail_after_chunks`` injects."""


@dataclass(frozen=True)
class Piece:
    """One token piece on a renderer-defined channel of one turn part."""

    part: int
    channel: str
    text: str


def limit_pieces(
    pieces: tuple[Piece, ...],
    budget: int | None,
) -> tuple[tuple[Piece, ...], bool]:
    """Apply an output-token budget; report whether it cut the output."""

    if budget is None or len(pieces) <= budget:
        return pieces, False
    return pieces[:budget], True


def chunk_pieces(
    pieces: Sequence[Piece],
    chunk_tokens: int | None,
) -> tuple[tuple[Piece, ...], ...]:
    """Group consecutive pieces of one part into stream chunks."""

    chunks: list[tuple[Piece, ...]] = []
    start = 0
    for end in range(1, len(pieces) + 1):
        full = chunk_tokens is not None and end - start == chunk_tokens
        if end == len(pieces) or full or pieces[end].part != pieces[start].part:
            chunks.append(tuple(pieces[start:end]))
            start = end
    return tuple(chunks)


def channel_text(chunk: Sequence[Piece], channel: str) -> str:
    return "".join(piece.text for piece in chunk if piece.channel == channel)


def chunk_delay_s(turn: Turn, chunk_index: int) -> float:
    """The scripted latency before streamed chunk ``chunk_index``."""

    return turn.first_token_delay_s if chunk_index == 0 else turn.inter_chunk_delay_s


def unary_delay_s(turn: Turn, chunk_count: int) -> float:
    """The latency a stream of ``chunk_count`` chunks would take in total."""

    return turn.first_token_delay_s + turn.inter_chunk_delay_s * max(0, chunk_count - 1)


async def sleep_s(delay: float) -> None:
    if delay > 0:
        await asyncio.sleep(delay)


def fails_before(turn: Turn, chunk_index: int, chunk_count: int) -> bool:
    """Whether the scripted failure fires before chunk ``chunk_index``.

    ``chunk_index == chunk_count`` is the terminal position (after the last
    chunk, before the final result), where an over-long budget fires.
    """

    limit = turn.fail_after_chunks
    if limit is None:
        return False
    return chunk_index == min(limit, chunk_count)
