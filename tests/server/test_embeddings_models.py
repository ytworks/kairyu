"""Embedding model registry routing, discovery, and bounded identity gates."""

import json

import pytest
from fastapi.testclient import TestClient

import kairyu.entrypoints.server.extra_routes as extra_routes_module
from kairyu.engine.embedding import EmbeddingCapacityError, EmbeddingResult
from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.server.app import create_app
from kairyu.entrypoints.server.extra_routes import MockEmbeddingBackend
from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.entrypoints.server.tenancy import TenantConfig, TenantLimits
from kairyu.orchestration.orchestrator import Orchestrator


class RecordingEmbeddingBackend(MockEmbeddingBackend):
    def __init__(self, dimensions: int) -> None:
        super().__init__(dimensions=dimensions)
        self.calls: list[tuple[str, ...]] = []

    async def embed(self, texts: list[str]) -> EmbeddingResult:
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


@pytest.mark.parametrize(
    "result",
    [
        EmbeddingResult(vectors=(), prompt_tokens=1, usage_exact=True),
        EmbeddingResult(vectors=((1.0, 2.0),), prompt_tokens=1, usage_exact=True),
        EmbeddingResult(
            vectors=((float("nan"), 0.0, 0.0, 0.0),),
            prompt_tokens=1,
            usage_exact=True,
        ),
        EmbeddingResult(
            vectors=((1.0, 0.0, 0.0, 0.0),),
            prompt_tokens=-1,
            usage_exact=True,
        ),
    ],
)
def test_malformed_embedding_backend_output_fails_closed(result):
    class MalformedBackend(MockEmbeddingBackend):
        async def embed(self, texts):
            return result

    app = create_app(
        {"chat": MockBackend()},
        embedding_backends={"embed": MalformedBackend(dimensions=4)},
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/embeddings",
            json={"model": "embed", "input": "hello"},
        )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "backend_error"
    assert "wrong" not in response.text
    assert "nan" not in response.text.lower()


def test_embedding_request_rejects_total_utf8_work_before_dispatch(monkeypatch):
    backend = RecordingEmbeddingBackend(dimensions=4)
    monkeypatch.setattr(
        extra_routes_module,
        "_MAX_EMBEDDING_UTF8_BYTES",
        5,
    )
    app = create_app(
        {"chat": MockBackend()},
        embedding_backends={"embed": backend},
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/embeddings",
            json={"model": "embed", "input": ["123", "456"]},
        )

    assert response.status_code == 400
    assert "UTF-8 bytes" in response.json()["error"]["message"]
    assert backend.calls == []


def test_embedding_capacity_rejection_is_429_and_refunds_predispatch_work():
    class BusyBackend(MockEmbeddingBackend):
        def __init__(self):
            super().__init__(dimensions=4)
            self.reports_exact_usage = True
            self.calls = 0

        def embed(self, texts):
            self.calls += 1
            raise EmbeddingCapacityError("private backend capacity detail")

    backend = BusyBackend()
    app = create_app(
        {"chat": MockBackend()},
        tenant_config=TenantConfig(
            limits={
                "default": TenantLimits(
                    requests_per_minute=60,
                    tokens_per_minute=100,
                    token_burst=100,
                )
            }
        ),
        embedding_backends={"embed": backend},
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/embeddings",
            json={"model": "embed", "input": "hello"},
        )

    assert response.status_code == 429
    assert response.headers["retry-after"] == "1"
    assert response.json()["error"] == {
        "message": "embedding backend is at max concurrency",
        "type": "rate_limit_error",
        "code": "embedding_concurrency_exceeded",
    }
    assert "private" not in response.text
    assert backend.calls == 1
    assert app.state.tenant_limiter.token_balance("default") == pytest.approx(100)
    assert app.state.tenant_limiter.reservation_snapshot()["default"] == 0
