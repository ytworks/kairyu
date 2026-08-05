"""MMLU: cheap zero-shot generated-letter accuracy without logprobs."""

from __future__ import annotations

import re
from collections import Counter

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

_EXPECTED_ROWS = 14_042
_EXPECTED_SUBJECTS = 57
_LETTERS = ("A", "B", "C", "D")
_CANONICAL_SUBJECTS = frozenset(
    {
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions",
    }
)
_GENERATED_ANSWER_RE = re.compile(
    r"\s*(?:answer\s*(?:is\s*)?:?\s*)?\(?([A-D])\)?[.\s]*\Z",
    re.IGNORECASE,
)


def extract_mmlu_choice(text: str) -> str | None:
    """Accept only a generated answer letter (optionally in a short marker)."""

    match = _GENERATED_ANSWER_RE.fullmatch(text)
    return match.group(1).upper() if match else None


def _prompt(payload: dict) -> str:
    subject = payload["subject"].replace("_", " ")
    choices = payload["choices"]
    return "\n".join(
        [
            f"The following is a multiple-choice question about {subject}.",
            "",
            payload["question"],
            *(f"{letter}. {choice}" for letter, choice in zip(_LETTERS, choices, strict=True)),
            "",
            "Answer with only the single letter A, B, C, or D.",
            "Answer:",
        ]
    )


class MmluAdapter(GenerativeAdapter):
    info = AdapterInfo(
        name="mmlu",
        display_name="MMLU",
        metric="zero-shot generated-letter accuracy",
        binary_outcomes=True,
        hf_dataset="cais/mmlu",
        annotations=(
            "zero-shot generated-letter adapter; canonical MMLU is 5-shot and "
            "selects A-D by next-token log probability, so scores are not comparable",
        ),
        comparable_to_published=False,
        incomparable_reason=(
            "Kairyu core uses zero-shot generated-letter exact match; canonical MMLU "
            "uses five demonstrations per subject and A-D next-token logprob argmax"
        ),
        evaluation_resources=(("kairyu.bench.adapters", "mmlu.py"),),
    )

    def normalize(self, ctx: DownloadContext) -> list[dict]:
        from kairyu.bench.hub import load_hf_rows

        rows = load_hf_rows(
            self.info.hf_dataset,
            name="all",
            split="test",
            revision=self.info.hf_revision,
        )
        if len(rows) != _EXPECTED_ROWS:
            raise DatasetUnavailable(
                f"{self.info.hf_dataset}@{self.info.hf_revision} all/test yielded "
                f"{len(rows)} rows, expected exactly {_EXPECTED_ROWS}"
            )

        per_subject: Counter[str] = Counter()
        normalized: list[dict] = []
        for index, row in enumerate(rows):
            question = row.get("question")
            subject = row.get("subject")
            choices = row.get("choices")
            answer = row.get("answer")
            if (
                not isinstance(question, str)
                or not question.strip()
                or not isinstance(subject, str)
                or not subject.strip()
                or not isinstance(choices, list)
                or len(choices) != 4
                or not all(
                    isinstance(choice, str) and bool(choice.strip()) for choice in choices
                )
                or isinstance(answer, bool)
                or not isinstance(answer, int)
                or not 0 <= answer < 4
            ):
                raise DatasetUnavailable(
                    f"MMLU row {index} must have question/subject strings, four string "
                    "choices, and an integer answer in [0, 3]"
                )
            item_index = per_subject[subject]
            per_subject[subject] += 1
            normalized.append(
                {
                    "id": f"mmlu-{subject}-{item_index:04d}",
                    "question": question,
                    "subject": subject,
                    "choices": list(choices),
                    "answer": _LETTERS[answer],
                }
            )

        subjects = frozenset(per_subject)
        if subjects != _CANONICAL_SUBJECTS:
            missing = sorted(_CANONICAL_SUBJECTS - subjects)
            extra = sorted(subjects - _CANONICAL_SUBJECTS)
            raise DatasetUnavailable(
                f"MMLU all/test subject set drifted (missing={missing}, extra={extra})"
            )
        return normalized

    def build_request(
        self, item: BenchItem, target: BenchTarget, ctx: RunContext
    ) -> ChatRequestSpec | SkipItem:
        return ChatRequestSpec(
            messages=({"role": "user", "content": _prompt(item.payload)},),
            max_tokens=min(target.max_output_tokens, 64),
        )

    async def score(
        self, item: BenchItem, response_text: str, ctx: RunContext
    ) -> ItemResult:
        extracted = extract_mmlu_choice(response_text)
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
                "config": "all",
                "split": "test",
                "expected_rows": _EXPECTED_ROWS,
                "expected_subjects": _EXPECTED_SUBJECTS,
                "shots": 0,
                "choice_order": "upstream A-D order; never shuffled",
                "response": "generated A-D letter exact match",
                "aggregation": "micro accuracy across test questions",
                "max_tokens": "min(target.max_output_tokens, 64)",
                "canonical_difference": (
                    "the original MMLU evaluates five-shot prompts and chooses among "
                    "the tokens ' A' through ' D' by logprob argmax"
                ),
            }
        )
        return methodology
