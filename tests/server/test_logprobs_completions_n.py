"""m9 D3 gates: logprobs surface, /v1/completions, n>1 — incl. OpenAI SDK."""

import httpx
import pytest

from kairyu.engine.core.sampler import Sampler
from kairyu.engine.core.torch_runner import TinyAttentionLM, TorchPagedRunner
from kairyu.engine.kairyu_backend import KairyuBackend
from kairyu.engine.tokenizer import ToyTokenizer
from kairyu.entrypoints.server.app import create_app


class _SmallVocabTokenizer(ToyTokenizer):
    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(t % 128 for t in super().encode(text))


def _real_backend(seed: int = 0) -> KairyuBackend:
    model = TinyAttentionLM(seed=seed)
    return KairyuBackend(
        num_pages=512,
        runner=TorchPagedRunner(model, num_pages=512, page_size=16, sampler=Sampler()),
        tokenizer=_SmallVocabTokenizer(),
    )


@pytest.fixture()
def app():
    return create_app(engines={"m": _real_backend()})


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


async def test_chat_logprobs_via_openai_sdk(app):
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key="x",
        base_url="http://t/v1",
        http_client=_client(app),
    )
    response = await client.chat.completions.create(
        model="m",
        messages=[{"role": "user", "content": "hello logprobs"}],
        max_tokens=4,
        logprobs=True,
        top_logprobs=3,
        temperature=0.0,
    )
    content = response.choices[0].logprobs.content
    assert len(content) == 4
    for entry in content:
        assert entry.logprob <= 0.0
        assert len(entry.top_logprobs) == 3
        assert entry.bytes == list(entry.token.encode())


async def test_top_logprobs_without_logprobs_is_400(app):
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "top_logprobs": 2,
            },
        )
    assert response.status_code == 400


async def test_streaming_logprobs_on_chunk_choice(app):
    import json as _json

    async with _client(app) as client:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "stream lp"}],
                "max_tokens": 3,
                "logprobs": True,
                "stream": True,
                "temperature": 0.0,
            },
        ) as response:
            entries = []
            async for line in response.aiter_lines():
                if not line.startswith("data:") or "[DONE]" in line:
                    continue
                chunk = _json.loads(line[len("data:") :])
                for choice in chunk["choices"]:
                    assert "logprobs" not in (choice.get("delta") or {})
                    if choice.get("logprobs"):
                        entries.extend(choice["logprobs"]["content"])
    assert len(entries) == 3  # one per generated token, no duplicates


async def test_legacy_completions_via_openai_sdk(app):
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key="x", base_url="http://t/v1", http_client=_client(app))
    response = await client.completions.create(
        model="m", prompt="legacy endpoint", max_tokens=4, logprobs=2, temperature=0.0
    )
    assert response.id.startswith("cmpl-")
    assert response.object == "text_completion"
    choice = response.choices[0]
    assert len(choice.logprobs.tokens) == 4
    assert len(choice.logprobs.token_logprobs) == 4
    assert len(choice.logprobs.text_offset) == 4
    assert choice.logprobs.text_offset[0] == 0
    assert response.usage.completion_tokens == 4


async def test_completions_rejects_unsupported_params(app):
    async with _client(app) as client:
        for field, value in (("echo", True), ("suffix", "s"), ("best_of", 2)):
            response = await client.post(
                "/v1/completions",
                json={"model": "m", "prompt": "x", field: value},
            )
            assert response.status_code == 400, field
        response = await client.post(
            "/v1/completions", json={"model": "m", "prompt": "x", "logprobs": 9}
        )
        assert response.status_code == 400


async def test_n_greater_than_one_distinct_and_indexed(app):
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "vary please"}],
                "max_tokens": 6,
                "n": 3,
                "temperature": 2.0,
            },
        )
    body = response.json()
    assert [c["index"] for c in body["choices"]] == [0, 1, 2]
    texts = {c["message"]["content"] for c in body["choices"]}
    assert len(texts) > 1  # unseeded n>1 at temperature>0 diverges
    # prompt counted once, completions summed (m9 D1 aggregation)
    usage = body["usage"]
    assert usage["completion_tokens"] == 18
    assert usage["total_tokens"] == usage["prompt_tokens"] + 18


async def test_seeded_n_is_reproducible(app):
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "seeded n"}],
        "max_tokens": 5,
        "n": 2,
        "temperature": 1.5,
        "seed": 11,
    }
    async with _client(app) as client:
        first = (await client.post("/v1/chat/completions", json=body)).json()
        second = (await client.post("/v1/chat/completions", json=body)).json()
    texts = lambda payload: [c["message"]["content"] for c in payload["choices"]]  # noqa: E731
    assert texts(first) == texts(second)


async def test_n_streaming_interleaves_indices(app):
    import json as _json

    async with _client(app) as client:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "stream n"}],
                "max_tokens": 4,
                "n": 2,
                "temperature": 2.0,
                "stream": True,
            },
        ) as response:
            indices = set()
            finish_indices = []
            async for line in response.aiter_lines():
                if not line.startswith("data:") or "[DONE]" in line:
                    continue
                chunk = _json.loads(line[len("data:") :])
                for choice in chunk["choices"]:
                    indices.add(choice["index"])
                    if choice.get("finish_reason"):
                        finish_indices.append(choice["index"])
    assert indices == {0, 1}
    assert sorted(finish_indices) == [0, 1]


async def test_zmq_backend_n_rejected_with_400():
    class _NoN:
        supports_n = False

        async def generate(self, request):  # pragma: no cover - never reached
            raise AssertionError

        def stream(self, request):  # pragma: no cover
            raise AssertionError

        async def shutdown(self):  # pragma: no cover
            return None

    app = create_app(engines={"z": _NoN()})
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "z", "messages": [{"role": "user", "content": "x"}], "n": 2},
        )
    assert response.status_code == 400
