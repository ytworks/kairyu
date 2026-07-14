"""Embedding model registry routing, discovery, and bounded identity gates."""

import json

import pytest
from fastapi.testclient import TestClient

from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.server.app import create_app
from kairyu.entrypoints.server.extra_routes import MockEmbeddingBackend
from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.orchestration.orchestrator import Orchestrator


class RecordingEmbeddingBackend(MockEmbeddingBackend):
    def __init__(self, dimensions: int) -> None:
        super().__init__(dimensions=dimensions)
        self.calls: list[tuple[str, ...]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(tuple(texts))
        return await super().embed(texts)


def test_unknown_embedding_model_is_rejected_before_backend_work():
    backend = RecordingEmbeddingBackend(dimensions=4)
    app = create_app(
        {"chat": MockBackend()},
        embedding_backends={"embed-small": backend},
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/embeddings",
            json={"model": "does-not-exist", "input": "hello"},
        )

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "message": "model 'does-not-exist' not found",
            "type": "invalid_request_error",
            "code": "model_not_found",
        }
    }
    assert backend.calls == []


def test_embedding_models_are_discoverable_and_route_independently():
    chat = MockBackend()
    small = RecordingEmbeddingBackend(dimensions=3)
    large = RecordingEmbeddingBackend(dimensions=7)
    app = create_app(
        {"chat": chat},
        orchestrators={"kairyu-auto": Orchestrator({"worker": chat})},
        embedding_backends={"embed-small": small, "embed-large": large},
    )

    with TestClient(app) as client:
        models = client.get("/v1/models")
        small_response = client.post(
            "/v1/embeddings",
            json={
                "model": "embed-small",
                "input": "small input",
                "encoding_format": "float",
            },
        )
        large_response = client.post(
            "/v1/embeddings",
            json={
                "model": "embed-large",
                "input": ["large one", "large two"],
                "encoding_format": "float",
            },
        )

    assert {model["id"] for model in models.json()["data"]} == {
        "chat",
        "kairyu-auto",
        "embed-small",
        "embed-large",
    }
    assert small_response.status_code == 200
    assert small_response.json()["model"] == "embed-small"
    assert len(small_response.json()["data"][0]["embedding"]) == 3
    assert large_response.status_code == 200
    assert large_response.json()["model"] == "embed-large"
    assert [len(item["embedding"]) for item in large_response.json()["data"]] == [
        7,
        7,
    ]
    assert small.calls == [("small input",)]
    assert large.calls == [("large one", "large two")]


def test_embedding_accounting_uses_only_resolved_model_identity(tmp_path):
    class RecordingLimiter:
        def __init__(self) -> None:
            self.charges: list[tuple[str, int]] = []

        def charge_tokens(self, tenant: str, tokens: int) -> None:
            self.charges.append((tenant, tokens))

    ledger_path = tmp_path / "usage.jsonl"
    backend = RecordingEmbeddingBackend(dimensions=4)
    app = create_app(
        {"chat": MockBackend()},
        settings=ServerSettings(usage_ledger_path=str(ledger_path)),
        embedding_backends={"embed-small": backend},
    )
    limiter = RecordingLimiter()
    app.state.tenant_limiter = limiter

    with TestClient(app) as client:
        successful = client.post(
            "/v1/embeddings",
            json={"model": "embed-small", "input": "two words"},
        )
        unknown = client.post(
            "/v1/embeddings",
            json={"model": "attacker-controlled", "input": "not billed"},
        )
        metrics = client.get("/metrics")

    assert successful.status_code == 200
    assert unknown.status_code == 404
    records = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert [record["model"] for record in records] == ["embed-small"]
    assert records[0]["prompt_tokens"] == 2
    assert records[0]["completion_tokens"] == 0
    assert limiter.charges == [("default", 2)]
    assert 'kairyu_requests_total{code="200",model="embed-small"} 1.0' in metrics.text
    assert 'kairyu_requests_total{code="404",model="unknown"} 1.0' in metrics.text
    assert "attacker-controlled" not in metrics.text


@pytest.mark.parametrize("surface", ["engine", "orchestrator"])
def test_create_app_rejects_embedding_model_collisions(surface):
    chat = MockBackend()
    engines = {"shared" if surface == "engine" else "chat": chat}
    orchestrators = (
        {"shared": Orchestrator({"worker": chat})}
        if surface == "orchestrator"
        else None
    )

    with pytest.raises(ValueError, match="collide"):
        create_app(
            engines,
            orchestrators=orchestrators,
            embedding_backends={
                "shared": RecordingEmbeddingBackend(dimensions=4)
            },
        )
