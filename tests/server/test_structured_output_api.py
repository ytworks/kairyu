"""m9 D4: response_format enforced end-to-end through the HTTP API."""

import json

import httpx
import pytest

from kairyu.engine.core.sampler import Sampler
from kairyu.engine.core.torch_runner import TinyAttentionLM, TorchPagedRunner
from kairyu.engine.kairyu_backend import KairyuBackend
from kairyu.entrypoints.server.app import create_app

# char-level vocab: printable ASCII + designated EOS (grammar terminates by
# sampling it, m8 D2)
_CHARS = [chr(c) for c in range(32, 127)] + ["<eos>"]
_EOS_ID = len(_CHARS) - 1


class _CharTokenizer:
    eos_token_id = _EOS_ID

    def encode(self, text: str) -> tuple[int, ...]:
        ids = tuple(ord(ch) - 32 for ch in text if 32 <= ord(ch) < 127)
        return ids or (0,)

    def decode(self, token_ids) -> str:
        return "".join(_CHARS[t] for t in token_ids if t != _EOS_ID)

    def vocab(self) -> list[str]:
        return _CHARS


def _backend() -> KairyuBackend:
    model = TinyAttentionLM(vocab=len(_CHARS), seed=0)
    runner = TorchPagedRunner(
        model, num_pages=512, page_size=16, sampler=Sampler(vocab_provider=lambda: _CHARS)
    )
    return KairyuBackend(num_pages=512, runner=runner, tokenizer=_CharTokenizer())


@pytest.fixture()
def app():
    return create_app(engines={"m": _backend()})


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


SCHEMAS = [
    {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
    {
        "type": "object",
        "properties": {"tag": {"type": "string", "enum": ["a", "b"]}},
        "required": ["tag"],
    },
]


@pytest.mark.parametrize("schema", SCHEMAS)
async def test_json_schema_yields_valid_json_and_stop(app, schema):
    pytest.importorskip("xgrammar")
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "emit json"}],
                "max_tokens": 100,
                "temperature": 0.0,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "s", "schema": schema},
                },
            },
        )
    assert response.status_code == 200
    choice = response.json()["choices"][0]
    parsed = json.loads(choice["message"]["content"])
    assert isinstance(parsed, dict)
    assert choice["finish_reason"] == "stop"  # grammar termination, not length


async def test_malformed_response_format_is_400_not_crash(app):
    async with _client(app) as client:
        bad_type = await client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "response_format": {"type": "bogus"},
            },
        )
        missing_schema = await client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "response_format": {"type": "json_schema"},
            },
        )
    assert bad_type.status_code == 400
    assert missing_schema.status_code == 400


async def test_response_format_text_passes_through(app):
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "plain"}],
                "max_tokens": 4,
                "response_format": {"type": "text"},
            },
        )
    assert response.status_code == 200
