import time

from kairyu.engine.mock import MockBackend
from kairyu.orchestration.moa import run_moa


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
