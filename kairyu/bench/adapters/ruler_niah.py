"""Deterministic RULER-style needle retrieval at fixed context lengths.

This is deliberately a small, package-owned NIAH probe, not an implementation
of the complete 13-task NVIDIA RULER suite.  One adapter instance represents
one point on the 4K--1M length curve, so the ordinary benchmark scoreboard,
history, and paired config comparison machinery remain the only reporting
path.
"""

from __future__ import annotations

import hashlib
import json

from kairyu.bench.adapters.base import (
    AdapterInfo,
    DownloadContext,
    GenerativeAdapter,
    RunContext,
    excerpt,
)
from kairyu.bench.types import (
    BenchItem,
    BenchTarget,
    ChatRequestSpec,
    DatasetUnavailable,
    ItemResult,
    SkipItem,
)

_SPECIFICATION = {
    "schema": "kairyu-ruler-style-niah-v1",
    "context_lengths": [
        4_096,
        8_192,
        16_384,
        32_768,
        65_536,
        131_072,
        262_144,
        524_288,
        1_048_576,
    ],
    "samples_per_length": 20,
    "encoding": "o200k_base",
    "output_tokens": 32,
    "filler": " grass",
    "identifier_namespace": "kairyu-ruler-niah-v1",
    "prompt": """Read the records below and remember the value assigned to each key.
When asked, answer with only the value, without explanation.

<records>{before}
The special record is: {key} = {value}.{after}
</records>

What value is assigned to {key}?
""",
}
CONTEXT_LENGTHS = tuple(_SPECIFICATION["context_lengths"])
SAMPLES_PER_LENGTH = _SPECIFICATION["samples_per_length"]
_ENCODING = _SPECIFICATION["encoding"]
_OUTPUT_TOKENS = _SPECIFICATION["output_tokens"]
_FILLER = _SPECIFICATION["filler"]
_PROMPT = _SPECIFICATION["prompt"]
_IDENTIFIER_NAMESPACE = _SPECIFICATION["identifier_namespace"]
_DATASET = "package:kairyu.bench.adapters/ruler_niah-spec"
_REVISION = "sha256:" + hashlib.sha256(
    json.dumps(
        _SPECIFICATION,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
).hexdigest()
_POSITIONS = tuple(
    (index + 0.5) / SAMPLES_PER_LENGTH for index in range(SAMPLES_PER_LENGTH)
)


def _encoder():
    try:
        import tiktoken
    except ImportError as error:  # pragma: no cover - exercised via monkeypatch
        raise DatasetUnavailable(
            "the long-context sweep needs tiktoken: pip install 'kairyu[bench]'"
        ) from error
    return tiktoken.get_encoding(_ENCODING)


def _identifier(context_tokens: int, index: int, kind: str) -> str:
    payload = f"{_IDENTIFIER_NAMESPACE}:{context_tokens}:{index}:{kind}".encode()
    return hashlib.sha256(payload).hexdigest()[:24]


def _render_prompt(
    encoder,
    *,
    context_tokens: int,
    index: int,
    position: float,
) -> tuple[str, str, float]:
    """Build one prompt with an exact ``o200k_base`` content-token length."""

    key = f"key-{_identifier(context_tokens, index, 'key')}"
    value = f"value-{_identifier(context_tokens, index, 'value')}"
    empty = _PROMPT.format(before="", after="", key=key, value=value)
    filler_count = context_tokens - len(encoder.encode(empty))
    if filler_count < 1:
        raise DatasetUnavailable(
            f"context length {context_tokens} cannot hold the NIAH prompt"
        )

    # `` grass`` is one stable o200k token, including when repeated.  Recount
    # the complete prompt rather than relying on that property silently: this
    # also fails closed if a future tokenizer revision changes segmentation.
    for _ in range(4):
        before_count = round(filler_count * position)
        prompt = _PROMPT.format(
            before=_FILLER * before_count,
            after=_FILLER * (filler_count - before_count),
            key=key,
            value=value,
        )
        measured = len(encoder.encode(prompt))
        if measured == context_tokens:
            needle_offset = prompt.index("\nThe special record is:")
            actual_position = len(encoder.encode(prompt[:needle_offset])) / context_tokens
            return prompt, value, actual_position
        filler_count += context_tokens - measured
        if filler_count < 1:
            break
    raise DatasetUnavailable(
        f"could not generate an exact {_ENCODING} {context_tokens}-token prompt"
    )


def ruler_niah_grade(response: str, answer: str) -> float:
    """Strict retrieval accuracy with surrounding whitespace ignored."""

    return 1.0 if response.strip() == answer else 0.0


class RulerNiahAdapter(GenerativeAdapter):
    """One fixed-length point in the Kairyu RULER-style NIAH curve."""

    def __init__(self, context_tokens: int) -> None:
        if context_tokens not in CONTEXT_LENGTHS:
            raise ValueError(f"unsupported NIAH context length: {context_tokens}")
        self.context_tokens = context_tokens
        label = f"{context_tokens // 1024}K"
        self.info = AdapterInfo(
            name=f"ruler-niah-{label.lower()}",
            display_name=f"RULER-style NIAH ({label})",
            metric="exact retrieval accuracy",
            hf_dataset=_DATASET,
            hf_revision=_REVISION,
            annotations=(
                "Kairyu's deterministic single-key NIAH probe; not the official "
                "13-task NVIDIA RULER suite",
                f"length is user-message content under {_ENCODING}; model chat-template "
                "tokens are endpoint-dependent",
            ),
            binary_outcomes=True,
        )

    def normalize(self, ctx: DownloadContext) -> list[dict]:
        del ctx
        encoder = _encoder()
        rows: list[dict] = []
        for index, position in enumerate(_POSITIONS):
            prompt, answer, actual_position = _render_prompt(
                encoder,
                context_tokens=self.context_tokens,
                index=index,
                position=position,
            )
            rows.append(
                {
                    "id": f"ruler-niah-{self.context_tokens}-{index:02d}",
                    "messages": [{"role": "user", "content": prompt}],
                    "answer": answer,
                    "context_tokens": self.context_tokens,
                    "needle_position": actual_position,
                    "target_needle_position": position,
                }
            )
        return rows

    def build_request(
        self, item: BenchItem, target: BenchTarget, ctx: RunContext
    ) -> ChatRequestSpec | SkipItem:
        del ctx
        if (
            target.max_context_tokens is not None
            and self.context_tokens > target.max_context_tokens
        ):
            return SkipItem(
                reason=(
                    f"curve point {self.context_tokens} tokens > target limit "
                    f"{target.max_context_tokens}"
                )
            )
        prompt_tokens = item.payload.get("context_tokens")
        if not isinstance(prompt_tokens, int) or prompt_tokens < 1:
            return SkipItem(reason="fixture/cache row has no valid context token count")
        return ChatRequestSpec(
            messages=tuple(item.payload["messages"]),
            max_tokens=min(target.max_output_tokens, _OUTPUT_TOKENS),
            est_prompt_tokens=prompt_tokens,
        )

    async def score(
        self, item: BenchItem, response_text: str, ctx: RunContext
    ) -> ItemResult:
        del ctx
        return ItemResult(
            item_id=item.id,
            status="completed",
            score=ruler_niah_grade(response_text, item.payload["answer"]),
            response_excerpt=excerpt(response_text),
        )

    def methodology(self, ctx: RunContext) -> dict:
        methodology = super().methodology(ctx)
        methodology.update(
            {
                "benchmark_family": "RULER-style",
                "task": "single-key needle retrieval in repeated-text haystack",
                "official_ruler_comparable": False,
                "dataset": _DATASET,
                "dataset_revision": _REVISION,
                "context_tokens": self.context_tokens,
                "tokenizer": _ENCODING,
                "token_scope": "user-message content; endpoint chat template excluded",
                "samples": SAMPLES_PER_LENGTH,
                "target_needle_positions": list(_POSITIONS),
                "max_tokens": f"min(target.max_output_tokens, {_OUTPUT_TOKENS})",
                "truncation_policy": (
                    "curve points above target.max_context_tokens are skipped, never truncated"
                ),
            }
        )
        return methodology


def ruler_niah_adapters() -> tuple[RulerNiahAdapter, ...]:
    return tuple(RulerNiahAdapter(length) for length in CONTEXT_LENGTHS)
