"""Cross-file contract of the Winnow-12B llama.cpp examples (LCP-D5)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from kairyu.deploy.spec import load_deployment_spec
from kairyu.engine.config_validation import validate_backend_options
from kairyu.entrypoints.server.app import _sse_chunk
from kairyu.entrypoints.server.protocol import ChunkDelta, ChunkToolCall, FunctionCall
from kairyu.entrypoints.server.sse_encode import ChatContentSSEEncoder

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _flag(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


@pytest.mark.parametrize("environment", ["winnow-12b-q8-1gpu", "winnow-12b-q8-dp8-8gpu"])
def test_l1_geometry_matches_what_kairyu_admits(environment):
    """Kairyu's per-request context and admission must be the slots
    winnow-server actually runs; a drift silently mis-sizes admission, and a
    default /readyz health URL would never mark a llama-server replica ready."""

    root = EXAMPLES / environment
    spec = json.loads((root / "example.json").read_text())
    runtime = spec["runtime"]
    compose = yaml.safe_load((root / "compose.yaml").read_text())
    deployment = load_deployment_spec((root / "kairyu.yaml").read_text())
    entries = (
        list(deployment.engines.values())
        or list(deployment.pools[spec["model"]["served_name"]].replicas)
    )

    assert len(entries) == len(spec["replicas"])
    for replica, entry in zip(spec["replicas"], entries, strict=True):
        validate_backend_options(entry.backend, entry.options)
        assert entry.options["upstream"] == "llamacpp"
        assert entry.options["max_model_len"] == runtime["slot_context_tokens"]
        assert entry.resolved_health_url() == f"http://{replica['service']}:8091/health"
        command = compose["services"][replica["service"]]["command"]
        slots = int(_flag(command, "--chat-parallel"))
        assert slots == runtime["chat_slots"]
        assert int(_flag(command, "--context")) == slots * runtime["slot_context_tokens"]
        sampling = runtime["sampling_defaults"]
        for flag, name in (
            ("--temp", "temperature"),
            ("--top-k", "top_k"),
            ("--top-p", "top_p"),
            ("--min-p", "min_p"),
            ("--repeat-penalty", "repeat_penalty"),
        ):
            assert float(_flag(command, flag)) == sampling[name]
    assert deployment.server.max_concurrency == runtime["chat_slots"] * len(spec["replicas"])


def _verification(environment: str):
    path = EXAMPLES / environment / "verification.py"
    spec = importlib.util.spec_from_file_location(f"verification_{environment}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("environment", ["winnow-12b-q8-1gpu", "winnow-12b-q8-dp8-8gpu"])
def test_attest_fails_when_server_sampling_defaults_are_missing(
    environment, tmp_path, monkeypatch
):
    """An absent default compared as NaN once passed the tolerance check, so
    attest verified LCP-D5 without seeing a single sampling default."""

    verification = _verification(environment)
    runtime = verification.RUNTIME

    def props(params: dict) -> dict:
        return {
            "build_info": f"b11036-{runtime['llama_cpp_commit'][:7]}",
            "total_slots": runtime["chat_slots"],
            "default_generation_settings": {
                "n_ctx": runtime["slot_context_tokens"],
                "params": params,
            },
            "chat_template_caps": {"supports_tool_calls": True},
            "modalities": {"vision": True},
        }

    def serve(params: dict) -> None:
        def fake_json(method, url, payload=None):
            if url.endswith("/props"):
                return 200, props(params)
            return 200, {"data": [{"id": verification.MODEL}]}

        monkeypatch.setattr(verification, "_http", lambda *args, **kwargs: (200, ""))
        monkeypatch.setattr(verification, "_json", fake_json)

    serve(dict(runtime["sampling_defaults"]))
    assert verification.attest(tmp_path) == 0
    serve({})
    assert verification.attest(tmp_path) == 1
    report = json.loads((tmp_path / "attest.json").read_text())
    for replica in verification.REPLICAS:
        assert "default temperature" in report["cases"][replica["service"]]


def _control(environment: str):
    path = EXAMPLES / environment / "control.py"
    spec = importlib.util.spec_from_file_location(f"control_{environment}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("environment", ["winnow-12b-q8-1gpu", "winnow-12b-q8-dp8-8gpu"])
def test_low_disk_blocks_only_start(environment, monkeypatch):
    """The free-space floor once ran for every action, so after the model
    download filled the disk `run.sh down` exited before calling Docker."""

    control = _control(environment)
    docker_calls: list[list[str]] = []
    monkeypatch.setattr(control.Path, "mkdir", lambda self, **kwargs: None)
    monkeypatch.setattr(
        control.shutil, "disk_usage", lambda path: SimpleNamespace(free=29 * 1024**3)
    )
    monkeypatch.setattr(
        control.subprocess,
        "run",
        lambda command, **kwargs: docker_calls.append(command),
    )
    for action in ("down", "status", "logs"):
        monkeypatch.setattr("sys.argv", ["control.py", action])
        control.main()
    assert [command[:2] for command in docker_calls] == [["docker", "compose"]] * 3
    monkeypatch.setattr("sys.argv", ["control.py", "up"])
    with pytest.raises(SystemExit, match="29 GiB free"):
        control.main()
    assert len(docker_calls) == 3


@pytest.mark.parametrize("environment", ["winnow-12b-q8-1gpu", "winnow-12b-q8-dp8-8gpu"])
def test_stream_gate_requires_an_assembled_tool_call(environment):
    """Kairyu's plain text chunks also carry `"tool_calls": null`, so a key
    search once passed the stream gate for a reply without any tool call."""

    verification = _verification(environment)
    model = verification.MODEL

    def finish(reason: str) -> str:
        return _sse_chunk("chatcmpl-1", 0, model, 0, ChunkDelta(), reason)

    done = "data: [DONE]\n\n"
    text = ChatContentSSEEncoder("chatcmpl-1", 0, model, include_usage=False)
    text_only = text.encode(0, "The weather is sunny.").decode() + finish("stop") + done
    call = ChunkToolCall(
        index=0,
        id="call_1",
        function=FunctionCall(name="get_weather", arguments='{"city": "Tokyo"}'),
    )
    tool_call = (
        _sse_chunk("chatcmpl-1", 0, model, 0, ChunkDelta(role="assistant", tool_calls=[call]))
        + finish("tool_calls")
        + done
    )
    assert verification._stream_tool_call_error(200, text_only, "get_weather") is not None
    assert verification._stream_tool_call_error(200, tool_call, "get_weather") is None
