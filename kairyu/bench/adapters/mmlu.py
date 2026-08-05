"""MMLU: zero-shot A-D ranking by teacher-forced continuation likelihood."""

from __future__ import annotations

from collections import Counter

from kairyu.bench.adapters.base import (
    AdapterInfo,
    DownloadContext,
    LogLikelihoodAdapter,
    RunContext,
    estimate_tokens,
)
from kairyu.bench.types import (
    BenchItem,
    BenchTarget,
    DatasetUnavailable,
    ItemResult,
    LogLikelihoodRequestSpec,
    LogLikelihoodResponse,
    SkipItem,
)

_EXPECTED_ROWS = 14_042
_EXPECTED_SUBJECTS = 57
_LETTERS = ("A", "B", "C", "D")
_CONTINUATIONS = tuple(f" {letter}" for letter in _LETTERS)
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
            "Answer:",
        ]
    )


class MmluAdapter(LogLikelihoodAdapter):
    info = AdapterInfo(
        name="mmlu",
        display_name="MMLU",
        metric="zero-shot continuation-likelihood accuracy",
        binary_outcomes=True,
        hf_dataset="cais/mmlu",
        annotations=(
            "zero-shot continuation-likelihood adapter; canonical MMLU is 5-shot, "
            "so scores are not comparable",
        ),
        comparable_to_published=False,
        incomparable_reason=(
            "Kairyu core uses zero-shot A-D continuation likelihood; canonical MMLU "
            "uses five demonstrations per subject"
        ),
        evaluation_resources=(
            ("kairyu.bench.adapters", "mmlu.py"),
            ("kairyu.bench.adapters", "base.py"),
            ("kairyu.bench", "types.py"),
        ),
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
    ) -> LogLikelihoodRequestSpec | SkipItem:
        context = _prompt(item.payload)
        estimated = estimate_tokens(context)
        if (
            target.max_context_tokens is not None
            and estimated > target.max_context_tokens
        ):
            return SkipItem(
                reason=(
                    f"est. {estimated} prompt tokens > target limit "
                    f"{target.max_context_tokens}"
                )
            )
        return LogLikelihoodRequestSpec(
            context=context,
            continuations=_CONTINUATIONS,
            reduction="sum",
            est_prompt_tokens=estimated,
        )

    async def score(
        self, item: BenchItem, response: LogLikelihoodResponse, ctx: RunContext
    ) -> ItemResult:
        candidate_details = [
            {
                "label": label,
                "text": item.payload["choices"][index],
                "continuation": candidate.continuation,
                "token_ids": list(candidate.token_ids),
                "tokens": list(candidate.tokens),
                "token_logprobs": list(candidate.token_logprobs),
                "text_offset": list(candidate.text_offsets),
                "sum_logprob": candidate.sum_logprob,
                "score": candidate.score,
            }
            for index, (label, candidate) in enumerate(
                zip(_LETTERS, response.candidates, strict=False)
            )
        ]
        evidence_details = {
            "prompt_token_ids": list(response.prompt_token_ids),
            "candidates": candidate_details,
        }
        if (
            response.reduction != "sum"
            or len(response.candidates) != len(_CONTINUATIONS)
            or tuple(
                candidate.continuation for candidate in response.candidates
            )
            != _CONTINUATIONS
        ):
            return ItemResult(
                item_id=item.id,
                status="failed",
                error="MMLU log-likelihood response changed candidate order or reduction",
                details=evidence_details,
            )
        if any(len(candidate.token_ids) != 1 for candidate in response.candidates):
            return ItemResult(
                item_id=item.id,
                status="failed",
                error=(
                    "MMLU requires each of ' A' through ' D' to tokenize as exactly "
                    "one continuation token"
                ),
                details=evidence_details,
            )
        candidate_token_ids = [
            candidate.token_ids[0] for candidate in response.candidates
        ]
        if len(set(candidate_token_ids)) != len(candidate_token_ids):
            return ItemResult(
                item_id=item.id,
                status="failed",
                error=(
                    "MMLU requires ' A' through ' D' to map to four distinct "
                    "continuation token IDs"
                ),
                details=evidence_details,
            )

        scores = [candidate.score for candidate in response.candidates]
        best_score = max(scores)
        tie_indices = [index for index, score in enumerate(scores) if score == best_score]
        winner_index = tie_indices[0]
        winner = _LETTERS[winner_index]
        runner_up_score = sorted(scores, reverse=True)[1]
        margin = best_score - runner_up_score
        return ItemResult(
            item_id=item.id,
            status="completed",
            score=1.0 if winner == item.payload["answer"] else 0.0,
            details={
                **evidence_details,
                "winner": winner,
                "gold": item.payload["answer"],
                "ties": [_LETTERS[index] for index in tie_indices],
                "margin": margin,
                "reduction": response.reduction,
            },
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
                "response": (
                    "rank exact continuations ' A' through ' D' by teacher-forced "
                    "selected-token log likelihood"
                ),
                "continuation_tokens": (
                    "exactly one distinct token ID per candidate or item failed"
                ),
                "reduction": "sum of selected-token logprobs",
                "tie_break": "stable A-D candidate order",
                "aggregation": "micro accuracy across test questions",
                "context_gate": (
                    "skip, never truncate, when chars/4 estimated prompt tokens exceed "
                    "target.max_context_tokens"
                ),
                "canonical_difference": (
                    "the original MMLU evaluates five-shot prompts; this adapter has "
                    "no subject demonstrations"
                ),
            }
        )
        return methodology

    def probe_result_fatal_reason(self, result: ItemResult) -> str | None:
        """Stop once the fixed A-D tokenizer structure is proven incompatible."""

        if (
            result.status == "failed"
            and result.error is not None
            and result.error.startswith("MMLU requires")
        ):
            return (
                "target tokenizer cannot represent MMLU's fixed A-D candidates "
                "as four distinct single tokens"
            )
        return None
