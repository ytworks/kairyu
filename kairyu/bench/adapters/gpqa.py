"""GPQA Diamond: 198 graduate-level MCQs (gated HF dataset), exact-match accuracy."""

from __future__ import annotations

from kairyu.bench.adapters.base import (
    AdapterInfo,
    DownloadContext,
    GenerativeAdapter,
    RunContext,
    excerpt,
    extract_choice_letter,
    mcq_prompt,
    shuffle_choices,
)
from kairyu.bench.hub import load_hf_rows
from kairyu.bench.types import BenchItem, BenchTarget, ChatRequestSpec, ItemResult, SkipItem


class GpqaDiamondAdapter(GenerativeAdapter):
    info = AdapterInfo(
        name="gpqa-diamond",
        display_name="GPQA Diamond",
        metric="accuracy",
        hf_dataset="Idavidrein/gpqa",
        gated=True,
    )

    def normalize(self, ctx: DownloadContext) -> list[dict]:
        rows = load_hf_rows(
            self.info.hf_dataset,
            name="gpqa_diamond",
            split="train",
            gated=True,
        )
        return [
            {
                "id": f"gpqa-diamond-{index:04d}",
                "question": row["Question"],
                "correct_answer": row["Correct Answer"].strip(),
                "incorrect_answers": [
                    row["Incorrect Answer 1"].strip(),
                    row["Incorrect Answer 2"].strip(),
                    row["Incorrect Answer 3"].strip(),
                ],
            }
            for index, row in enumerate(rows)
        ]

    def build_request(
        self, item: BenchItem, target: BenchTarget, ctx: RunContext
    ) -> ChatRequestSpec | SkipItem:
        payload = item.payload
        choices, _ = shuffle_choices(
            ctx.seed, item.id, payload["correct_answer"], payload["incorrect_answers"]
        )
        prompt = mcq_prompt(payload["question"], choices)
        return ChatRequestSpec(
            messages=({"role": "user", "content": prompt},),
            max_tokens=target.max_output_tokens,
        )

    async def score(
        self, item: BenchItem, response_text: str, ctx: RunContext
    ) -> ItemResult:
        payload = item.payload
        _, correct_letter = shuffle_choices(
            ctx.seed, item.id, payload["correct_answer"], payload["incorrect_answers"]
        )
        extracted = extract_choice_letter(response_text)
        return ItemResult(
            item_id=item.id,
            status="completed",
            score=1.0 if extracted == correct_letter else 0.0,
            response_excerpt=excerpt(response_text),
        )
