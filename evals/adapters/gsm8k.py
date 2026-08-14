"""GSM8K: zero-shot numeric exact-match accuracy without a judge model."""

from __future__ import annotations

import re

from evals.adapters.base import (
    AdapterInfo,
    DownloadContext,
    GenerativeAdapter,
    RunContext,
    excerpt,
)
from evals.types import (
    BenchItem,
    BenchTarget,
    ChatRequestSpec,
    DatasetUnavailable,
    ItemResult,
    SkipItem,
)

_EXPECTED_ROWS = 1_319
_ANSWER_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
_PROMPT = """Solve the following grade-school math problem. Show your reasoning, then end
your response with exactly one final line in the form `#### <number>`.

{question}"""


def extract_gsm8k_answer(text: str) -> str | None:
    """Apply the original GSM8K marker parser and comma-only normalization."""

    match = _ANSWER_RE.search(text)
    if match is None:
        return None
    return match.group(1).strip().replace(",", "")


class Gsm8kAdapter(GenerativeAdapter):
    info = AdapterInfo(
        name="gsm8k",
        display_name="GSM8K",
        metric="zero-shot exact-match accuracy",
        binary_outcomes=True,
        hf_dataset="openai/gsm8k",
        annotations=(
            "zero-shot chat prompt; the original GSM8K release specifies the "
            "#### numeric scorer but not this prompt",
        ),
        evaluation_resources=(("evals.adapters", "gsm8k.py"),),
    )

    def normalize(self, ctx: DownloadContext) -> list[dict]:
        from evals.hub import load_hf_rows

        rows = load_hf_rows(
            self.info.hf_dataset,
            name="main",
            split="test",
            revision=self.info.hf_revision,
        )
        if len(rows) != _EXPECTED_ROWS:
            raise DatasetUnavailable(
                f"{self.info.hf_dataset}@{self.info.hf_revision} main/test yielded "
                f"{len(rows)} rows, expected exactly {_EXPECTED_ROWS}"
            )

        normalized: list[dict] = []
        for index, row in enumerate(rows):
            question = row.get("question")
            source_answer = row.get("answer")
            if (
                not isinstance(question, str)
                or not question.strip()
                or not isinstance(source_answer, str)
                or not source_answer.strip()
            ):
                raise DatasetUnavailable(
                    f"GSM8K row {index} must contain string question/answer fields"
                )
            answer = extract_gsm8k_answer(source_answer)
            if answer is None:
                raise DatasetUnavailable(
                    f"GSM8K row {index} has no official #### numeric answer"
                )
            normalized.append(
                {
                    "id": f"gsm8k-test-{index:04d}",
                    "question": question,
                    "answer": answer,
                }
            )
        return normalized

    def build_request(
        self, item: BenchItem, target: BenchTarget, ctx: RunContext
    ) -> ChatRequestSpec | SkipItem:
        return ChatRequestSpec(
            messages=(
                {"role": "user", "content": _PROMPT.format(question=item.payload["question"])},
            ),
            max_tokens=min(target.max_output_tokens, 1024),
        )

    async def score(
        self, item: BenchItem, response_text: str, ctx: RunContext
    ) -> ItemResult:
        extracted = extract_gsm8k_answer(response_text)
        return ItemResult(
            item_id=item.id,
            status="completed",
            score=1.0 if extracted == item.payload["answer"] else 0.0,
            response_excerpt=excerpt(response_text),
        )

    def methodology(self, ctx: RunContext) -> dict:
        methodology = super().methodology(ctx)
        methodology.update(
            {
                "config": "main",
                "split": "test",
                "expected_rows": _EXPECTED_ROWS,
                "prompt": "zero-shot reasoning; final line must be #### <number>",
                "scorer": (
                    r"first /#### (\-?[0-9\.\,]+)/ match; remove commas; "
                    "string exact match; no numeric tolerance"
                ),
                "max_tokens": "min(target.max_output_tokens, 1024)",
            }
        )
        return methodology
