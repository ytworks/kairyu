import importlib.util

import pytest

from kairyu import SamplingParams
from kairyu.engine.backend import CacheHint, GenerationRequest
from kairyu.engine.mock import MockBackend


def _request(prompt: str, params: SamplingParams | None = None) -> GenerationRequest:
    return GenerationRequest(
        request_id="req-1",
        prompt=prompt,
        sampling_params=params or SamplingParams(),
    )


async def test_generate_is_deterministic_echo():
    backend = MockBackend()
    first = await backend.generate(_request("hello world"))
    second = await backend.generate(_request("hello world"))
    assert first.completions[0].text == second.completions[0].text
    assert "hello world" in first.completions[0].text
    assert first.finished is True


async def test_canned_response_matches_substring():
    backend = MockBackend(responses={"[verifier]": "PASS: looks good"})
    result = await backend.generate(_request("[verifier] check this answer"))
    assert result.completions[0].text == "PASS: looks good"


async def test_n_samples_produce_distinct_completions():
    backend = MockBackend()
    result = await backend.generate(_request("hi", SamplingParams(n=3)))
    assert len(result.completions) == 3
    assert [c.index for c in result.completions] == [0, 1, 2]
    texts = {c.text for c in result.completions}
    assert len(texts) == 3


async def test_stream_chunks_reassemble_to_full_text():
    backend = MockBackend()
    request = _request("stream me please this is a long prompt")
    full = await backend.generate(request)
    streamed = []
    finished_flags = []
    async for partial in backend.stream(request):
        streamed.append(partial.completions[0].text)
        finished_flags.append(partial.finished)
    assert streamed[-1] == full.completions[0].text
    assert finished_flags[-1] is True
    assert all(flag is False for flag in finished_flags[:-1])
    assert len(streamed) > 1


async def test_cache_hint_travels_with_request():
    hint = CacheHint(session_id="s1", prefix_fingerprint="abc")
    request = GenerationRequest(
        request_id="r", prompt="p", sampling_params=SamplingParams(), cache_hint=hint
    )
    assert request.cache_hint.session_id == "s1"


async def test_call_log_records_prompts():
    backend = MockBackend()
    await backend.generate(_request("first"))
    await backend.generate(_request("second"))
    assert [p for p in backend.prompts_seen] == ["first", "second"]


def test_tensor_parallel_size_recorded_for_plumbing_assertions():
    assert MockBackend().tensor_parallel_size == 1
    assert MockBackend(tensor_parallel_size=4).tensor_parallel_size == 4


def test_tensor_parallel_size_below_one_rejected():
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        MockBackend(tensor_parallel_size=0)


def test_llm_default_backend_forwards_tensor_parallel_size(monkeypatch):
    from kairyu.entrypoints import llm as llm_module

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    backend = llm_module._default_backend("m", None, tensor_parallel_size=3)
    assert isinstance(backend, MockBackend)
    assert backend.tensor_parallel_size == 3


def test_llm_constructor_forwards_tensor_parallel_size(monkeypatch):
    from kairyu.entrypoints.llm import LLM

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    llm = LLM(model="m", tensor_parallel_size=2)
    assert isinstance(llm.backend, MockBackend)
    assert llm.backend.tensor_parallel_size == 2


def test_async_engine_default_backend_forwards_tensor_parallel_size(monkeypatch):
    from kairyu.entrypoints import async_engine

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    args = async_engine.AsyncEngineArgs(model="m", tensor_parallel_size=2)
    backend = async_engine._default_backend(args)
    assert isinstance(backend, MockBackend)
    assert backend.tensor_parallel_size == 2
