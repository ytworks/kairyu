"""CPU-only contracts for the pinned G4 M-A3 Kairyu launcher."""

from __future__ import annotations

import asyncio
import json

import pytest

from verification.l1.performance import g4_ma3_kairyu_server as server


class _Backend:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.shutdown_calls = 0

    async def shutdown(self) -> None:
        self.shutdown_calls += 1


def _expected_backend_kwargs(depth: int) -> dict[str, object]:
    return {
        "model_path": server.MODEL_PATH,
        "tokenizer": server.MODEL_PATH,
        "tensor_parallel_size": 1,
        "expert_parallel_size": 4,
        "expert_parallel_attention_dp": True,
        "pipeline_depth": depth,
        "decode_mode": "cuda_graph",
        "cuda_graph_max_batch": 32,
        "cuda_graph_max_pages": 128,
        "cuda_graph_warmup_iters": 3,
        "num_pages": 4096,
        "page_size": 16,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 256,
        "max_model_len": 2048,
        "priority_age_s": None,
        "kv_cache_dtype": "bfloat16",
    }


@pytest.mark.parametrize("depth", server.PIPELINE_DEPTHS)
def test_main_pins_backend_app_uvicorn_and_resolved_json(
    monkeypatch,
    capsys,
    depth: int,
) -> None:
    captured: dict[str, object] = {}

    class Backend(_Backend):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            captured["backend"] = self

    def fake_create_app(*, engines, settings, lifespan):
        captured["engines"] = engines
        captured["settings"] = settings
        captured["lifespan"] = lifespan
        return "formal-app"

    def fake_run(app, **kwargs) -> None:
        captured["uvicorn_app"] = app
        captured["uvicorn_kwargs"] = kwargs

    monkeypatch.setattr(server, "KairyuBackend", Backend)
    monkeypatch.setattr(server, "create_app", fake_create_app)
    monkeypatch.setattr(server.uvicorn, "run", fake_run)

    server.main(["--pipeline-depth", str(depth)])

    backend = captured["backend"]
    assert isinstance(backend, Backend)
    assert backend.kwargs == _expected_backend_kwargs(depth)
    assert captured["engines"] == {server.SERVED_MODEL_NAME: backend}
    settings = captured["settings"]
    assert settings.access_log is False
    assert captured["uvicorn_app"] == "formal-app"
    assert captured["uvicorn_kwargs"] == {
        "host": server.DEFAULT_HOST,
        "port": server.DEFAULT_PORT,
        "loop": "uvloop",
        "http": "httptools",
        "log_config": None,
        "access_log": False,
    }

    emitted = json.loads(capsys.readouterr().out)
    assert emitted == {
        "schema_version": server.SCHEMA_VERSION,
        "served_model_name": server.SERVED_MODEL_NAME,
        "host": server.DEFAULT_HOST,
        "port": server.DEFAULT_PORT,
        "access_log": False,
        **_expected_backend_kwargs(depth),
    }


def test_lifespan_always_shuts_down_backend(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class Backend(_Backend):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            captured["backend"] = self

    def fake_create_app(*, engines, settings, lifespan):
        del engines, settings
        captured["lifespan"] = lifespan
        return object()

    monkeypatch.setattr(server, "KairyuBackend", Backend)
    monkeypatch.setattr(server, "create_app", fake_create_app)
    args = server.build_parser().parse_args(["--pipeline-depth", "1"])
    server.build_application(args)

    async def exercise() -> None:
        lifespan = captured["lifespan"]
        with pytest.raises(RuntimeError, match="formal stop"):
            async with lifespan(object()):
                raise RuntimeError("formal stop")

    asyncio.run(exercise())
    backend = captured["backend"]
    assert isinstance(backend, Backend)
    assert backend.shutdown_calls == 1


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--pipeline-depth", "2"],
        ["--pipeline-depth", "5", "--model-path", "/tmp/model"],
        ["--pipeline-depth", "1", "--port", "0"],
        ["--pipeline-depth", "1", "--port", "65536"],
        ["--pipeline-depth", "1", "--host", ""],
        ["--pipeline-depth", "1", "--host", "bad host"],
        ["--pipeline-d", "1"],
    ],
)
def test_parser_fails_closed(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as stopped:
        server.build_parser().parse_args(argv)
    assert stopped.value.code == 2


def test_backend_kwargs_rejects_unreviewed_depth() -> None:
    with pytest.raises(ValueError, match="pipeline_depth"):
        server.backend_kwargs(2)
