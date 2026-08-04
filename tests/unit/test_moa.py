import time
from dataclasses import replace

import pytest

from kairyu.engine.backend import GenerationUsage
from kairyu.engine.mock import MockBackend
from kairyu.engine.prompt import TemplatedPrompt
from kairyu.orchestration.moa import run_moa, stream_moa


@pytest.mark.parametrize("streaming", [False, True])
async def test_moa_rejects_templated_query_before_derivation(streaming):
    backend = MockBackend()
    query = TemplatedPrompt("<BOS>user hello<ASSISTANT>")

    with pytest.raises(ValueError, match="cannot derive proposal prompts.*pre-rendered"):
        if streaming:
            _ = [event async for event in stream_moa(backend, query)]
        else:
            await run_moa(backend, query)

    assert backend.prompts_seen == ()


@pytest.mark.parametrize("streaming", [False, True])
async def test_moa_rejects_templated_shared_prefix(streaming):
    backend = MockBackend()
    prefix = TemplatedPrompt("<BOS>already rendered")

    with pytest.raises(ValueError, match="shared_prefix.*pre-rendered"):
        if streaming:
            _ = [event async for event in stream_moa(backend, "q", shared_prefix=prefix)]
        else:
            await run_moa(backend, "q", shared_prefix=prefix)

    assert backend.prompts_seen == ()


async def test_moa_collects_n_proposals_and_synthesizes():
    backend = MockBackend()
    result = await run_moa(backend, "explain KV caching", n_samples=3)
    assert len(result.proposals) == 3
    assert len(set(result.proposals)) == 3  # distinct seeds -> distinct samples
    assert result.final_text
    synthesis_prompt = backend.prompts_seen[-1]
    for proposal in result.proposals:
        assert proposal in synthesis_prompt


async def test_moa_proposals_run_concurrently():
    latency = 0.05
    backend = MockBackend(latency_s=latency)
    start = time.perf_counter()
    await run_moa(backend, "q", n_samples=4)
    elapsed = time.perf_counter() - start
    assert elapsed < latency * 4  # 4 proposals overlap + 1 synthesis call


async def test_moa_uses_separate_synthesizer_backend_when_given():
    proposer = MockBackend()
    synthesizer = MockBackend(responses={"Synthesize": "final answer"})
    result = await run_moa(proposer, "q", n_samples=2, synthesizer=synthesizer)
    assert result.final_text == "final answer"
    assert len(proposer.prompts_seen) == 2
    assert len(synthesizer.prompts_seen) == 1


async def test_moa_prompts_share_prefix():
    backend = MockBackend()
    prefix = "SYS\n"
    await run_moa(backend, "q", n_samples=2, shared_prefix=prefix)
    assert all(p.startswith(prefix) for p in backend.prompts_seen)


async def test_moa_sums_cached_tokens_across_proposals_and_synthesis():
    class CachedBackend(MockBackend):
        async def generate(self, request):
            result = await super().generate(request)
            return replace(
                result,
                usage=GenerationUsage(
                    prompt_tokens=10,
                    completion_tokens=2,
                    cached_tokens=4,
                ),
            )

    result = await run_moa(CachedBackend(), "q", n_samples=3)

    assert result.usage == (40, 8)
    assert result.cached_tokens == 16


async def test_moa_stream_pulls_synthesizer_deltas_and_preserves_usage():
    class CachedBackend(MockBackend):
        async def generate(self, request):
            result = await super().generate(request)
            return replace(
                result,
                usage=GenerationUsage(
                    prompt_tokens=10,
                    completion_tokens=2,
                    cached_tokens=4,
                ),
            )

    events = [
        event
        async for event in stream_moa(
            CachedBackend(
                responses={
                    "Synthesize": "a final answer long enough for multiple chunks"
                }
            ),
            "q",
            n_samples=2,
        )
    ]

    assert [event.kind for event in events[:-1]] == ["delta"] * (len(events) - 1)
    result = events[-1].result
    assert result is not None
    assert "".join(event.text for event in events[:-1]) == result.final_text
    assert result.usage == (30, 6)
    assert result.cached_tokens == 12
