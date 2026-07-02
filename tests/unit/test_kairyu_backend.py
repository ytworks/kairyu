import asyncio

import httpx

from kairyu import SamplingParams
from kairyu.engine.backend import GenerationRequest
from kairyu.engine.kairyu_backend import KairyuBackend
from kairyu.engine.registry import create_backend
from kairyu.entrypoints.server.app import create_app


def _request(request_id: str, prompt: str, max_tokens: int = 4) -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        prompt=prompt,
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )


async def test_generate_runs_through_engine_core():
    backend = KairyuBackend(num_pages=256)
    result = await backend.generate(_request("r1", "hello world from kairyu"))
    assert result.finished is True
    completion = result.completions[0]
    assert len(completion.token_ids) == 4  # max_tokens honored by the scheduler
    assert completion.text
    assert completion.finish_reason == "length"


async def test_stream_yields_incremental_partials():
    backend = KairyuBackend(num_pages=256)
    partials = []
    async for partial in backend.stream(_request("r2", "stream me", max_tokens=5)):
        partials.append(partial)
    assert len(partials) >= 5
    assert partials[-1].finished is True
    assert all(p.finished is False for p in partials[:-1])
    lengths = [len(p.completions[0].token_ids) for p in partials]
    assert lengths == sorted(lengths)  # monotonically growing


async def test_concurrent_requests_are_continuously_batched():
    backend = KairyuBackend(num_pages=256)
    results = await asyncio.gather(
        backend.generate(_request("a", "first prompt")),
        backend.generate(_request("b", "second prompt")),
        backend.generate(_request("c", "third prompt")),
    )
    assert all(r.finished for r in results)
    assert {r.request_id for r in results} == {"a", "b", "c"}


def test_registered_as_kairyu_backend():
    backend = create_backend("kairyu", num_pages=64)
    assert isinstance(backend, KairyuBackend)


async def test_full_stack_openai_server_over_engine_core():
    app = create_app(engines={"kairyu-cpu": KairyuBackend(num_pages=256)})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "kairyu-cpu",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4,
            },
        )
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"]
    assert data["choices"][0]["finish_reason"] == "length"
