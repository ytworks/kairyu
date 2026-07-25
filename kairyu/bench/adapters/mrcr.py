"""MRCRv2 (OpenAI multi-round co-reference): string-similarity scoring, no judge.

Official metric: the response must start with the per-item random prepend
string; the score is SequenceMatcher.ratio() between response and answer
(both with the prepend stripped), else 0.

The published `openai/mrcr` split mixes 2-, 4- and 8-needle items across
context lengths up to 1M tokens. Fugu reports the **8-needle** subset at up to
**128K** context, so this adapter selects exactly that slice: averaging over the
whole 2,400-row split would compare a different (easier, shorter) population
against Fugu's number.
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher

from kairyu.bench.adapters.base import (
    AdapterInfo,
    DownloadContext,
    GenerativeAdapter,
    RunContext,
    estimate_tokens,
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

# Fugu's reported conditions for this row.
_NEEDLES = 8
_MAX_CONTEXT_TOKENS = 131_072  # "up to 128K context"


def prompt_tokens(messages: list[dict], n_chars: object = None) -> int:
    """Estimated prompt tokens for an MRCR row.

    The dataset's own `n_chars` is preferred when present (it counts the whole
    conversation upstream); otherwise the shared chars/4 heuristic is summed
    over the messages. Both are estimates and recorded as such.
    """
    if isinstance(n_chars, int | float) and n_chars > 0:
        return int(n_chars) // 4 + 1
    return sum(estimate_tokens(str(message.get("content", ""))) for message in messages)


def mrcr_grade(response: str, answer: str, prepend: str) -> float:
    if not response.startswith(prepend):
        return 0.0
    response = response.removeprefix(prepend)
    answer = answer.removeprefix(prepend)
    return SequenceMatcher(None, response, answer).ratio()


class MrcrAdapter(GenerativeAdapter):
    info = AdapterInfo(
        name="mrcr-v2",
        display_name="MRCRv2",
        metric="sequence-match ratio",
        hf_dataset="openai/mrcr",
        annotations=(
            f"{_NEEDLES}-needle items at up to "
            f"{_MAX_CONTEXT_TOKENS // 1024}K estimated prompt tokens (Fugu's "
            "reported slice); the published split also contains 2/4-needle and "
            "longer-context items, which are excluded",
        ),
    )

    def normalize(self, ctx: DownloadContext) -> list[dict]:
        from kairyu.bench.hub import load_hf_rows

        rows = load_hf_rows(self.info.hf_dataset, split="train")
        normalized: list[dict] = []
        wrong_needles = too_long = 0
        for index, row in enumerate(rows):
            if row.get("n_needles") != _NEEDLES:
                wrong_needles += 1
                continue
            messages = json.loads(row["prompt"])
            est = prompt_tokens(messages, row.get("n_chars"))
            if est > _MAX_CONTEXT_TOKENS:
                too_long += 1
                continue
            normalized.append(
                {
                    "id": f"mrcr-{index:05d}",
                    "messages": messages,
                    "answer": row["answer"],
                    "prepend": row["random_string_to_prepend"],
                    "n_needles": row.get("n_needles"),
                    "est_prompt_tokens": est,
                }
            )
        if not normalized:
            raise DatasetUnavailable(
                f"{self.info.hf_dataset} has no {_NEEDLES}-needle item within "
                f"{_MAX_CONTEXT_TOKENS} estimated prompt tokens "
                f"({len(rows)} rows seen)"
            )
        # The excluded counts are the denominator story; never silent.
        print(
            f"[mrcr-v2] kept {len(normalized)}/{len(rows)} rows "
            f"({wrong_needles} not {_NEEDLES}-needle, {too_long} over "
            f"{_MAX_CONTEXT_TOKENS} estimated prompt tokens)"
        )
        return normalized

    def build_request(
        self, item: BenchItem, target: BenchTarget, ctx: RunContext
    ) -> ChatRequestSpec | SkipItem:
        messages = item.payload["messages"]
        est = item.payload.get("est_prompt_tokens")
        if not isinstance(est, int):
            est = prompt_tokens(messages)
        if target.max_context_tokens is not None and est > target.max_context_tokens:
            return SkipItem(
                reason=f"est. {est} prompt tokens > target limit {target.max_context_tokens}"
            )
        return ChatRequestSpec(
            messages=tuple(messages),
            max_tokens=target.max_output_tokens,
            est_prompt_tokens=est,
        )

    async def score(
        self, item: BenchItem, response_text: str, ctx: RunContext
    ) -> ItemResult:
        score = mrcr_grade(response_text, item.payload["answer"], item.payload["prepend"])
        return ItemResult(
            item_id=item.id,
            status="completed",
            score=score,
            response_excerpt=excerpt(response_text),
        )

    def methodology(self, ctx: RunContext) -> dict:
        base = super().methodology(ctx)
        base["needles"] = _NEEDLES
        base["max_context_tokens"] = _MAX_CONTEXT_TOKENS
        base["selection"] = (
            f"only n_needles == {_NEEDLES} rows whose estimated prompt tokens "
            f"(dataset n_chars/4 when present, else chars/4 over messages) are "
            f"<= {_MAX_CONTEXT_TOKENS}"
        )
        base["truncation_policy"] = (
            "items whose ~chars/4 estimated prompt tokens exceed the target's "
            "max_context_tokens are skipped, never truncated"
        )
        return base
