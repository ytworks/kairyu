import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "examples/qwen3.8-deepseek-v4.1-8gpu/capacity.py"
spec = importlib.util.spec_from_file_location("v41_capacity", PATH)
capacity = importlib.util.module_from_spec(spec)


def test_retrieval_requires_stop_exact_key_and_usage():
    spec.loader.exec_module(capacity)
    result = {
        "finish_reason": "stop",
        "content": "K123",
        "prompt_tokens": 8200,
        "completion_tokens": 5,
    }
    assert capacity.retrieval_passed(result, "K123", 8192, 8192, 1048576)
    for field, value in [
        ("finish_reason", "length"),
        ("content", "The key is K123"),
        ("prompt_tokens", 1),
    ]:
        assert not capacity.retrieval_passed(
            dict(result, **{field: value}), "K123", 8192, 8192, 1048576
        )


def test_fixed_budget_is_not_completed_answer_gate():
    spec.loader.exec_module(capacity)
    result = {
        "finish_reason": "length",
        "completion_tokens": 256,
        "content": "",
        "reasoning_content": "private",
        "prompt_tokens": 8200,
    }
    assert capacity.fixed_passed(result, 256, 8192)
    assert not capacity.retrieval_passed(result, "K123", 8192, 8192, 1048576)
    assert not capacity.fixed_passed(dict(result, completion_tokens=255), 256, 8192)


def test_runtime_guard_rejects_tp8_and_wrong_devices():
    spec.loader.exec_module(capacity)
    record = {
        "Image": "sha256:pin",
        "State": {"Running": True},
        "Config": {
            "Env": [
                "KAIRYU_REQUIREMENTS_CONFIG_SHA256="
                + capacity.control._requirements_config_sha256()
            ],
            "Cmd": capacity.expected_runtime_command(),
        },
        "HostConfig": {"DeviceRequests": [{"DeviceIDs": list(map(str, range(6)))}]},
    }
    assert capacity.validate_runtime(record, "sha256:pin")
    record["State"]["Running"] = False
    assert not capacity.validate_runtime(record, "sha256:pin")
    record["State"]["Running"] = True
    tp_index = record["Config"]["Cmd"].index("--tensor-parallel-size") + 1
    record["Config"]["Cmd"][tp_index] = "8"
    assert not capacity.validate_runtime(record, "sha256:pin")
    record["Config"]["Cmd"][tp_index] = "2"
    record["HostConfig"]["DeviceRequests"][0]["DeviceIDs"] = ["0", "1"]
    assert not capacity.validate_runtime(record, "sha256:pin")


def test_stream_keeps_raw_evidence_and_missing_done_fails(tmp_path):
    import asyncio
    import json

    import httpx

    spec.loader.exec_module(capacity)

    async def run(terminal):
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "think"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "K123"}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 8200, "completion_tokens": 12}},
        ]
        text = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
        if terminal:
            text += "data: [DONE]\n\n"
        transport = httpx.MockTransport(lambda request: httpx.Response(200, text=text))
        async with httpx.AsyncClient(transport=transport, base_url="http://test/v1/") as client:
            directory = tmp_path / str(terminal)
            result = await capacity.request(client, {"model": "test"}, directory, 1)
            assert json.loads((directory / "request.json").read_text()) == {"model": "test"}
            assert "K123" in (directory / "response.sse").read_text()
            assert result["transport_passed"] is terminal
            if terminal:
                assert result["model_ttft_ms"] <= result["content_ttft_ms"]
                assert result["reasoning_content"] == "think"
        return result

    asyncio.run(run(True))
    asyncio.run(run(False))


def test_runtime_guard_rejects_stale_startup_configuration():
    spec.loader.exec_module(capacity)
    record = {
        "Image": "sha256:pin",
        "State": {"Running": True},
        "Config": {
            "Env": ["KAIRYU_REQUIREMENTS_CONFIG_SHA256=stale"],
            "Cmd": [
                "model",
                "--tensor-parallel-size",
                "2",
                "--data-parallel-size",
                "3",
                "--enable-expert-parallel",
            ],
        },
        "HostConfig": {"DeviceRequests": [{"DeviceIDs": list(map(str, range(6)))}]},
    }
    assert not capacity.validate_runtime(record, "sha256:pin")


def test_runtime_identity_does_not_copy_private_environment():
    spec.loader.exec_module(capacity)
    record = {
        "Id": "container-id",
        "Name": "/ds6",
        "Image": "sha256:pin",
        "State": {"Running": True, "StartedAt": "start"},
        "Config": {"Env": ["HF_TOKEN=private", "KAIRYU_REQUIREMENTS_CONFIG_SHA256=digest"]},
    }
    identity = capacity.runtime_identity(record)
    assert identity["config_sha256"] == "digest"
    assert identity["container_id"] == "container-id"
    assert "private" not in str(identity) and "HF_TOKEN" not in str(identity)


def test_endpoint_must_be_published_by_attested_container():
    spec.loader.exec_module(capacity)
    record = {
        "NetworkSettings": {"Ports": {"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8009"}]}}
    }
    assert capacity.endpoint_matches(record, "http://127.0.0.1:8009/v1")
    assert not capacity.endpoint_matches(record, "http://127.0.0.1:8010/v1")
    assert not capacity.endpoint_matches(record, "http://other-host:8009/v1")
    assert not capacity.endpoint_matches(record, "http://127.0.0.1:8009/other/v1")


def test_runtime_guard_rejects_overridden_context_limit_with_current_hash():
    import yaml

    spec.loader.exec_module(capacity)
    command = yaml.safe_load((PATH.parent / "compose.yaml").read_text())["services"]["deepseek"][
        "command"
    ]
    record = {
        "Image": "sha256:pin",
        "State": {"Running": True},
        "Config": {
            "Cmd": command,
            "Env": [
                "KAIRYU_REQUIREMENTS_CONFIG_SHA256="
                + capacity.control._requirements_config_sha256()
            ],
        },
        "HostConfig": {"DeviceRequests": [{"DeviceIDs": list(map(str, range(6)))}]},
    }
    assert capacity.validate_runtime(record, "sha256:pin")
    command[command.index("--max-model-len") + 1] = "32768"
    assert not capacity.validate_runtime(record, "sha256:pin")


def test_native_reasoning_alias_counts_as_model_output(tmp_path):
    import asyncio
    import json

    import httpx

    spec.loader.exec_module(capacity)

    async def run():
        chunks = [
            {"choices": [{"delta": {"reasoning": "private"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "length"}]},
            {"choices": [], "usage": {"prompt_tokens": 8222, "completion_tokens": 256}},
        ]
        raw = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n"
        transport = httpx.MockTransport(lambda request: httpx.Response(200, text=raw))
        async with httpx.AsyncClient(transport=transport, base_url="http://test/v1/") as client:
            directory = tmp_path / "native"
            result = await capacity.request(client, {}, directory, 1)
            assert result["transport_passed"]
            assert result["reasoning_content"] == "private"
            assert result["content"] == ""
            assert result["content_ttft_ms"] is None
            assert result["model_ttft_ms"] is not None
            assert capacity.fixed_passed(result, 256, 8192)
            assert (directory / "response.sse").read_text() == raw

    asyncio.run(run())


def test_reasoning_alias_precedes_content_timing_without_double_count(monkeypatch):
    import asyncio
    import json

    spec.loader.exec_module(capacity)
    clock = iter([1.0, 2.0, 3.0, 4.0])
    monkeypatch.setattr(capacity.benchmark.time, "perf_counter", lambda: next(clock))

    async def lines():
        for delta, finish in [
            ({"reasoning": "first"}, None),
            ({"reasoning_content": "second", "reasoning": "second"}, None),
            ({"content": "answer"}, "stop"),
        ]:
            yield "data: " + json.dumps({"choices": [{"delta": delta, "finish_reason": finish}]})
        yield 'data: {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}'
        yield "data: [DONE]"

    result = asyncio.run(capacity.benchmark.collect(capacity.normalize_reasoning_alias(lines()), 0))
    assert result["reasoning_content"] == "firstsecond"
    assert result["content"] == "answer"
    assert result["model_ttft_ms"] == 1000
    assert result["content_ttft_ms"] == 3000
    assert result["tpot_ms"] == 750
