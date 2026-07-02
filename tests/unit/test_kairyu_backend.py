import asyncio

import httpx
import pytest

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


def test_tensor_parallel_size_recorded():
    backend = KairyuBackend(num_pages=64, tensor_parallel_size=2)
    assert backend.tensor_parallel_size == 2


@pytest.mark.parametrize("degree", [0, 3])
def test_tensor_parallel_size_rejects_invalid_degrees(degree):
    with pytest.raises(ValueError):
        KairyuBackend(num_pages=64, tensor_parallel_size=degree)


async def test_tp2_greedy_output_equals_tp1():
    # Deterministic toy ranks + FakeCommunicator: TP must not change sampling
    # (design m5 D1: same step, identical samples on every rank).
    prompts = ["hello world from kairyu", "another prompt entirely"]
    results = {}
    for degree in (1, 2):
        backend = KairyuBackend(num_pages=256, tensor_parallel_size=degree)
        outputs = await asyncio.gather(
            *(
                backend.generate(_request(f"tp{degree}-{i}", prompt, max_tokens=5))
                for i, prompt in enumerate(prompts)
            )
        )
        results[degree] = [result.completions[0].token_ids for result in outputs]
    assert results[1] == results[2]


def test_registry_forwards_tensor_parallel_size():
    backend = create_backend("kairyu", num_pages=64, tensor_parallel_size=4)
    assert isinstance(backend, KairyuBackend)
    assert backend.tensor_parallel_size == 4


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
