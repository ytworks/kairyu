"""LLM judge over any OpenAI-compatible endpoint (may be the kairyu gateway itself).

Verdicts are parsed from a strict `correct: yes|no` field; anything
unparseable yields correct=None so the item is recorded "unjudged" — a judge
can degrade a run, never crash it.
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass

from evals.judge_prompts import judge_template
from evals.types import JudgeConfig, JudgeEndpointConfig

# allow markdown emphasis around the label and verdict — a judge that writes
# "**correct:** yes" or "Correct: **Yes**" must still parse (M10)
_VERDICT_RE = re.compile(r"(?im)^\s*\**\s*correct\s*\**\s*:\s*\**\s*(yes|no)\b")
_EXCERPT = 500
# bound the graded response fed into the judge prompt: an uncapped 200k-char
# response would become a 200k-token judge call (cost) and widen the prompt
# injection surface (M7). Generous enough not to trip on normal answers.
_RESPONSE_CAP = 32_000
_JUDGE_MAX_TOKENS = 4096  # reasoning judges need room before the verdict line (M10)
_JUDGE_TEMPERATURE = 0.0
_JUDGE_TIMEOUT_S = 300.0
JUDGE_PROTOCOL_VERSION = 1


def judge_protocol_identity() -> dict:
    """Scoring-protocol fields whose changes invalidate judged evidence."""
    return {
        "version": JUDGE_PROTOCOL_VERSION,
        "verdict_pattern": _VERDICT_RE.pattern,
        "verdict_flags": _VERDICT_RE.flags,
        "response_cap": _RESPONSE_CAP,
        "evidence_excerpt_cap": _EXCERPT,
        "judge_max_tokens": _JUDGE_MAX_TOKENS,
        "temperature": _JUDGE_TEMPERATURE,
        "timeout_s": _JUDGE_TIMEOUT_S,
    }


@dataclass(frozen=True)
class JudgeVerdict:
    correct: bool | None
    raw_excerpt: str
    model: str
    votes: tuple[JudgeVerdict, ...] = ()
    vote_identities: tuple[tuple[str, str], ...] = ()
    aggregation: str | None = None
    failure_reason: str | None = None

    def as_dict(self) -> dict:
        evidence = {
            "model": self.model,
            "correct": self.correct,
            "raw_excerpt": self.raw_excerpt,
        }
        # Preserve the established three-key artifact for a single judge.
        if self.votes:
            recorded_votes = []
            for index, vote in enumerate(self.votes):
                recorded = vote.as_dict()
                if index < len(self.vote_identities):
                    recorded["base_url"] = self.vote_identities[index][0]
                recorded_votes.append(recorded)
            evidence.update(
                {
                    "aggregation": self.aggregation,
                    "votes": recorded_votes,
                }
            )
            if self.failure_reason is not None:
                evidence["failure_reason"] = self.failure_reason
        return evidence

    def constituent_verdicts(self) -> tuple[JudgeVerdict, ...]:
        return self.votes or (self,)


def parse_verdict(text: str) -> bool | None:
    matches = _VERDICT_RE.findall(text)
    if not matches:
        return None
    return matches[-1].lower() == "yes"


class JudgeClient:
    def __init__(self, config: JudgeEndpointConfig, *, http_factory) -> None:
        if isinstance(config, JudgeConfig) and config.additional_judges:
            raise ValueError(
                "JudgeClient cannot ignore additional_judges; use "
                "build_judge_client()"
            )
        if not config.enabled:
            raise ValueError("JudgeClient requires judge.base_url and judge.model")
        self.config = config
        self._http_factory = http_factory
        self._semaphore = asyncio.Semaphore(config.concurrency)

    async def grade(
        self,
        *,
        question: str,
        expected: str,
        response: str,
        template_name: str = "hle",
        counterbalanced: bool = False,
        template: str | None = None,
    ) -> JudgeVerdict:
        from evals.adapters.base import (
            RequestFailed,
            call_chat,
        )
        from evals.types import BenchTarget, ChatRequestSpec

        # ``template`` is retained for direct-library backward compatibility.
        # SuiteRunner adapters never use it: their registered selector is what
        # the run fingerprint binds. Calibration also stays registry-backed.
        prompt_template = (
            template
            if template is not None
            else judge_template(template_name, counterbalanced=counterbalanced)
        )
        prompt = prompt_template.format(
            question=question, expected=expected, response=response[:_RESPONSE_CAP]
        )
        target = BenchTarget(
            base_url=self.config.base_url,
            model=self.config.model,
            api_key_env=self.config.api_key_env,
            reasoning_effort=self.config.reasoning_effort,
            top_p=self.config.top_p,
            seed=self.config.seed,
            extra_body_json=self.config.extra_body_json,
        )
        request = ChatRequestSpec(
            messages=({"role": "user", "content": prompt},),
            max_tokens=_JUDGE_MAX_TOKENS,
            temperature=_JUDGE_TEMPERATURE,
        )
        api_key = os.environ.get(self.config.api_key_env)
        async with self._semaphore:
            try:
                async with self._http_factory() as client:
                    text = await call_chat(
                        client,
                        target,
                        request,
                        retries=self.config.max_retries,
                        timeout_s=_JUDGE_TIMEOUT_S,
                        api_key=api_key,
                    )
            except RequestFailed as error:
                return JudgeVerdict(
                    correct=None,
                    raw_excerpt=f"judge request failed: {error}"[:_EXCERPT],
                    model=self.config.model,
                )
            except Exception as error:  # noqa: BLE001 - degradation is evidence
                return JudgeVerdict(
                    correct=None,
                    raw_excerpt=(
                        f"judge request raised {type(error).__name__}: {error}"
                    )[:_EXCERPT],
                    model=self.config.model,
                )
        return JudgeVerdict(
            correct=parse_verdict(text),
            raw_excerpt=text[:_EXCERPT],
            model=self.config.model,
        )


class MajorityJudgeClient:
    """Strict-majority panel for pointwise grading.

    Every configured member must return a parseable vote. A missing vote or a
    tie yields ``correct=None`` so adding a hedge can never silently collapse
    back to a single judge. For two total members, this is unanimous consensus.
    ``config`` deliberately remains the primary JudgeConfig because τ-bench
    consumes it as its one user-simulator endpoint.
    """

    def __init__(self, config: JudgeConfig, *, http_factory) -> None:
        endpoints = config.grading_endpoints()
        if len(endpoints) < 2:
            raise ValueError("MajorityJudgeClient requires at least two judges")
        self.config = config
        self._members = tuple(
            JudgeClient(endpoint, http_factory=http_factory) for endpoint in endpoints
        )

    @property
    def members(self) -> tuple[JudgeClient, ...]:
        return self._members

    async def grade(self, **kwargs) -> JudgeVerdict:
        gathered = await asyncio.gather(
            *(member.grade(**kwargs) for member in self._members),
            return_exceptions=True,
        )
        votes: list[JudgeVerdict] = []
        for member, result in zip(self._members, gathered, strict=True):
            if isinstance(result, Exception):
                votes.append(
                    JudgeVerdict(
                        correct=None,
                        raw_excerpt=f"judge raised {type(result).__name__}: {result}"[
                            :_EXCERPT
                        ],
                        model=member.config.model,
                    )
                )
            else:
                votes.append(result)

        unavailable = [vote.model for vote in votes if vote.correct is None]
        yes = sum(vote.correct is True for vote in votes)
        no = sum(vote.correct is False for vote in votes)
        threshold = len(votes) // 2 + 1
        failure_reason = None
        if unavailable:
            correct = None
            failure_reason = "unavailable vote(s): " + ", ".join(unavailable)
        elif yes >= threshold:
            correct = True
        elif no >= threshold:
            correct = False
        else:
            correct = None
            failure_reason = f"no strict majority ({yes} yes, {no} no)"

        summary = failure_reason or (
            f"strict majority ({yes} yes, {no} no; threshold {threshold})"
        )
        return JudgeVerdict(
            correct=correct,
            raw_excerpt=summary,
            model=f"{self.config.aggregation}-panel",
            votes=tuple(votes),
            vote_identities=tuple(
                member.config.resolved_identity() for member in self._members
            ),
            aggregation=self.config.aggregation,
            failure_reason=failure_reason,
        )


def build_judge_client(config: JudgeConfig, *, http_factory):
    """Build the compatibility single client or an explicitly requested panel."""
    if config.additional_judges:
        return MajorityJudgeClient(config, http_factory=http_factory)
    return JudgeClient(config, http_factory=http_factory)
