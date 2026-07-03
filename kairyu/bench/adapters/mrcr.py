"""MRCRv2 (OpenAI multi-round co-reference): string-similarity scoring, no judge.

Official metric: the response must start with the per-item random prepend
string; the score is SequenceMatcher.ratio() between response and answer
(both with the prepend stripped), else 0.
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
from kairyu.bench.types import BenchItem, BenchTarget, ChatRequestSpec, ItemResult, SkipItem


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
    )

    def normalize(self, ctx: DownloadContext) -> list[dict]:
        from kairyu.bench.hub import load_hf_rows

        rows = load_hf_rows(self.info.hf_dataset, split="train")
        return [
            {
                "id": f"mrcr-{index:05d}",
                "messages": json.loads(row["prompt"]),
                "answer": row["answer"],
                "prepend": row["random_string_to_prepend"],
                "n_needles": row.get("n_needles"),
            }
            for index, row in enumerate(rows)
        ]

    def build_request(
        self, item: BenchItem, target: BenchTarget, ctx: RunContext
    ) -> ChatRequestSpec | SkipItem:
        messages = item.payload["messages"]
        est = sum(estimate_tokens(str(m.get("content", ""))) for m in messages)
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
        base["truncation_policy"] = (
            "items whose ~chars/4 estimated prompt tokens exceed the target's "
            "max_context_tokens are skipped, never truncated"
        )
        return base
