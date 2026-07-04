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

from kairyu.bench.judge_prompts import HLE_JUDGE_TEMPLATE
from kairyu.bench.types import JudgeConfig

_VERDICT_RE = re.compile(r"(?im)^\s*correct\s*:\s*\**\s*(yes|no)\b")
_EXCERPT = 500


@dataclass(frozen=True)
class JudgeVerdict:
    correct: bool | None
    raw_excerpt: str
    model: str

    def as_dict(self) -> dict:
        return {"model": self.model, "correct": self.correct, "raw_excerpt": self.raw_excerpt}


def parse_verdict(text: str) -> bool | None:
    matches = _VERDICT_RE.findall(text)
    if not matches:
        return None
    return matches[-1].lower() == "yes"


class JudgeClient:
    def __init__(self, config: JudgeConfig, *, http_factory) -> None:
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
        template: str = HLE_JUDGE_TEMPLATE,
    ) -> JudgeVerdict:
        from kairyu.bench.adapters.base import (
            RequestFailed,
            call_chat,
        )
        from kairyu.bench.types import BenchTarget, ChatRequestSpec

        prompt = template.format(question=question, expected=expected, response=response)
        target = BenchTarget(
            base_url=self.config.base_url,
            model=self.config.model,
            api_key_env=self.config.api_key_env,
        )
        request = ChatRequestSpec(
            messages=({"role": "user", "content": prompt},), max_tokens=2048
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
                        timeout_s=300.0,
                        api_key=api_key,
                    )
            except RequestFailed as error:
                return JudgeVerdict(
                    correct=None,
                    raw_excerpt=f"judge request failed: {error}"[:_EXCERPT],
                    model=self.config.model,
                )
        return JudgeVerdict(
            correct=parse_verdict(text),
            raw_excerpt=text[:_EXCERPT],
            model=self.config.model,
        )
