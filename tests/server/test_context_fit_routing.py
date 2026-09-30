"""A public AUTO model never routes a conversation to a worker that cannot
hold it. The profile judge reads a bounded view of the latest turn, so on
main a 300K-token conversation was judged onto a 262K-context route and the
upstream 400 surfaced as a 502 (PR #602, Issue #599).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from kairyu.engine.backend import GenerationResult, GenerationUsage
from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.orchestration.conductor import Conductor, RoleSpec
from kairyu.orchestration.orchestrator import (
    EngineDescriptor,
    Orchestrator,
    ProfileChoice,
    ProfileJudge,
)
from kairyu.orchestration.router import RuleRouter
from kairyu.outputs import CompletionOutput
from kairyu.sampling_params import SamplingParams
from tests.server._legacy_chat import create_legacy_app

COMPLEX = (
    "First research the options and summarize trade-offs. Then design a plan. "
    "After that implement it. Finally verify everything end to end."
)


class SizedBackend:
    """A worker with a declared context that counts 4 characters per token."""

    def __init__(self, name: str, max_model_len: int) -> None:
        self.name = name
        self.max_model_len = max_model_len
        self.prompts: list[str] = []
        self.counted = 0

    async def count_prompt_tokens_async(self, prompt: str) -> int:
        self.counted += 1
        return len(prompt) // 4

    async def generate(self, request):
        prompt = str(request.prompt)
        self.prompts.append(prompt)
        text = "SMALL" if "Reply with exactly one word" in prompt else f"{self.name}-answer"
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(CompletionOutput(index=0, text=text, token_ids=(), finish_reason="stop"),),
            usage=GenerationUsage(prompt_tokens=1, completion_tokens=1),
        )

    async def stream(self, request):
        yield await self.generate(request)

    async def shutdown(self) -> None:
        return None


class MultiAgentRouter:
    def preview(self, query, context=None):
        return RuleRouter().route(COMPLEX)

    def route(self, query, context=None):
        return RuleRouter().route(COMPLEX)


def _app(tmp_path, small: SizedBackend, large: SizedBackend):
    primary = (
        RoleSpec(name="draft", worker="small", prompt="[draft] {query}"),
        RoleSpec(
            name="final",
            worker="large",
            role_type="synthesizer",
            depends_on=("draft",),
            prompt="[final] {query} {draft}",
        ),
    )
    # The long-conversation twin: the small worker reads a bounded digest.
    primary_long = (
        RoleSpec(name="brief", worker="large", prompt="[brief] {query}"),
        RoleSpec(name="draft", worker="small", depends_on=("brief",), prompt="[draft] {brief}"),
        RoleSpec(
            name="final",
            worker="large",
            role_type="synthesizer",
            depends_on=("draft",),
            prompt="[final] {query} {draft}",
        ),
    )
    small_direct = (
        RoleSpec(name="direct", worker="small", role_type="publisher", prompt="[direct] {query}"),
    )
    orchestrator = Orchestrator(
        {"small": small, "large": large},
        router=MultiAgentRouter(),
        roles=primary,
        profiles={"primary_long": primary_long, "small_direct": small_direct},
        profile_judge=ProfileJudge(
            worker="small",
            choices=(
                ProfileChoice("small_direct", "SMALL", "easy requests"),
                ProfileChoice("primary", "ENSEMBLE", "hard requests"),
            ),
        ),
        context_fallbacks={"primary": "primary_long"},
        sampling_params=SamplingParams(max_tokens=1024),
        engine_descriptors={
            "small": EngineDescriptor("mock", "small"),
            "large": EngineDescriptor("mock", "large"),
        },
    )
    return create_legacy_app(
        {},
        orchestrators={"auto": orchestrator},
        settings=ServerSettings(usage_ledger_path=str(tmp_path / "usage.jsonl")),
    )


def _chat(client: TestClient, text: str):
    return client.post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": [{"role": "user", "content": text}], "max_tokens": 512},
    )


def test_short_conversation_is_judged_as_before(tmp_path):
    small, large = SizedBackend("small", 16_384), SizedBackend("large", 200_000)
    with TestClient(_app(tmp_path, small, large)) as client:
        response = _chat(client, "hello there")

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "small-answer"
    # The byte bound already proved every route fits: no exact count.
    assert small.counted == large.counted == 0


def test_long_conversation_avoids_routes_its_workers_cannot_hold(tmp_path):
    small, large = SizedBackend("small", 16_384), SizedBackend("large", 200_000)
    long_text = "word " * 16_000  # 80,000 chars = 20,000 tokens > the small context
    with TestClient(_app(tmp_path, small, large)) as client:
        response = _chat(client, long_text)

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "large-answer"
    # No judge call (one route left) and the small worker never saw the
    # conversation: it only drafted from the large worker's brief.
    assert small.prompts and all(long_text not in prompt for prompt in small.prompts)
    assert not any("Reply with exactly one word" in prompt for prompt in small.prompts)
    assert any(prompt.startswith("[brief]") for prompt in large.prompts)


def test_conversation_no_route_can_hold_is_rejected_before_dispatch(tmp_path):
    small, large = SizedBackend("small", 16_384), SizedBackend("large", 50_000)
    with TestClient(_app(tmp_path, small, large)) as client:
        response = _chat(client, "word " * 60_000)  # 75,000 tokens

    assert response.status_code == 400
    assert "context_length_exceeded" in response.json()["error"]["message"]
    assert small.prompts == large.prompts == []


async def test_head_that_cannot_fit_is_skipped_and_the_final_answers_headless():
    small, large = SizedBackend("small", 16_384), SizedBackend("large", 200_000)
    roles = (
        RoleSpec(name="head", worker="small", role_type="head", prompt="[head] {query}"),
        RoleSpec(
            name="final",
            worker="large",
            role_type="synthesizer",
            depends_on=("head",),
            prompt="[final] continue {head}",
            prompt_headless="[final-headless] {query}",
        ),
    )
    conductor = Conductor(
        roles=roles,
        workers={"small": small, "large": large},
        final_sampling_params=SamplingParams(max_tokens=512),
        context_tokens={"small": (20_000, 20_000), "large": (20_000, 20_000)},
    )
    result = await conductor.run("q")

    assert small.prompts == []
    assert large.prompts == ["[final-headless] q"]
    assert any(event.kind == "skipped:context" for event in result.trace)


def test_an_uncountable_conversation_is_served_not_rejected(tmp_path):
    class Uncountable(SizedBackend):
        async def count_prompt_tokens_async(self, prompt: str) -> None:
            return None

    small, large = Uncountable("small", 16_384), Uncountable("large", 50_000)
    with TestClient(_app(tmp_path, small, large)) as client:
        response = _chat(client, "word " * 60_000)

    # Unknown size never becomes a 400 on its own; the upstream decides.
    assert response.status_code == 200
