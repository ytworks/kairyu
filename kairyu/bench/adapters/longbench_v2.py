""""Long Context Reasoning" slot — LongBench v2 substitute.

Fugu's own "Long Context Reasoning" row is an unpublished internal suite;
this slot runs the public LongBench v2 MCQ set instead and every report
carries the not-comparable annotation (user decision 4).
"""

from __future__ import annotations

from kairyu.bench.adapters.base import (
    AdapterInfo,
    DownloadContext,
    GenerativeAdapter,
    RunContext,
    estimate_tokens,
    excerpt,
    extract_choice_letter,
)
from kairyu.bench.types import BenchItem, BenchTarget, ChatRequestSpec, ItemResult, SkipItem

_PROMPT = """Read the context and answer the multiple-choice question.

<context>
{context}
</context>

Question: {question}

Choices:
A) {a}
B) {b}
C) {c}
D) {d}

Answer with the single letter of the correct choice. End your reply with "Answer: <letter>"."""


class LongBenchV2Adapter(GenerativeAdapter):
    info = AdapterInfo(
        name="long-context-reasoning",
        display_name="Long Context Reasoning",
        metric="accuracy",
        hf_dataset="THUDM/LongBench-v2",
        annotations=(
            "LongBench v2 substitute — Fugu's 'Long Context Reasoning' suite is "
            "unpublished; scores are NOT directly comparable to Fugu's number",
        ),
    )

    def normalize(self, ctx: DownloadContext) -> list[dict]:
        from kairyu.bench.hub import load_hf_rows

        rows = load_hf_rows(self.info.hf_dataset, split="train")
        return [
            {
                "id": row["_id"],
                "question": row["question"],
                "choices": [row["choice_A"], row["choice_B"], row["choice_C"], row["choice_D"]],
                "answer": row["answer"].strip().upper(),
                "context": row["context"],
                "length": row.get("length"),
                "domain": row.get("domain"),
            }
            for row in rows
        ]

    def build_request(
        self, item: BenchItem, target: BenchTarget, ctx: RunContext
    ) -> ChatRequestSpec | SkipItem:
        payload = item.payload
        prompt = _PROMPT.format(
            context=payload["context"],
            question=payload["question"],
            a=payload["choices"][0],
            b=payload["choices"][1],
            c=payload["choices"][2],
            d=payload["choices"][3],
        )
        est = estimate_tokens(prompt)
        if target.max_context_tokens is not None and est > target.max_context_tokens:
            return SkipItem(
                reason=f"est. {est} prompt tokens > target limit {target.max_context_tokens}"
            )
        return ChatRequestSpec(
            messages=({"role": "user", "content": prompt},),
            max_tokens=min(target.max_output_tokens, 2048),
            est_prompt_tokens=est,
        )

    async def score(
        self, item: BenchItem, response_text: str, ctx: RunContext
    ) -> ItemResult:
        extracted = extract_choice_letter(response_text)
        return ItemResult(
            item_id=item.id,
            status="completed",
            score=1.0 if extracted == item.payload["answer"] else 0.0,
            response_excerpt=excerpt(response_text),
        )

    def methodology(self, ctx: RunContext) -> dict:
        base = super().methodology(ctx)
        base["truncation_policy"] = (
            "items whose ~chars/4 estimated prompt tokens exceed the target's "
            "max_context_tokens are skipped, never truncated"
        )
        return base
