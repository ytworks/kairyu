"""V4.1's default/override and evidence contracts differ from the V4 examples."""

import ast
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/deepseek-v4.1-flash-8gpu"


def load(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_effort_patch_rejects_drift_and_is_idempotent():
    patch = load("patch_runtime")
    original = (
        "from typing import Dict\n"
        "REASONING_EFFORT_MAPPINGS: Dict[str, int] = "
        "{'low':25,'high':50,'xhigh':75,'max':100}\n"
    )
    changed = patch.align_efforts(original)
    # xhigh is L3 max on the Responses wire (m11 D4 amendment 2026-10-17).
    assert ast.literal_eval(ast.parse(changed).body[1].value) == {
        "low": 50,
        "high": 75,
        "xhigh": 100,
        "max": 100,
    }
    assert patch.align_efforts(changed) == changed
    with pytest.raises(ValueError, match="Unrecognized"):
        patch.align_efforts(original.replace("'high':50", "'high':60"))
    with pytest.raises(ValueError, match="Expected one"):
        patch.align_efforts("pass\n")


def test_runtime_edits_fail_closed_on_missing_or_duplicate_anchors():
    patch = load("patch_runtime")
    for original in ("unchanged", "anchor\nanchor\n"):
        with pytest.raises(ValueError, match="source drift"):
            patch.replace_once(original, "anchor\n", "anchor\naddition\n")
    changed = patch.replace_once("anchor\n", "anchor\n", "anchor\naddition\n")
    assert patch.replace_once(changed, "anchor\n", "anchor\naddition\n") == changed


@pytest.mark.parametrize(
    "capability,model,allowed",
    [
        (100, "deepseek_v4", True),
        (120, "deepseek_v41", True),
        (120, "deepseek_v4", False),
        (90, "deepseek_v41", False),
    ],
)
def test_mxfp4_enablement_is_limited_to_verified_model_and_device(capability, model, allowed):
    source = (
        "def guard(vllm_config):\n"
        "    use_fp4 = True\n"
        "    if use_fp4 and not current_platform.is_device_capability_family(100):\n"
        "        raise ValueError('unsupported')\n"
        "    return use_fp4\n"
    )
    namespace = {
        "current_platform": SimpleNamespace(
            is_device_capability_family=lambda family: family == capability
        )
    }
    exec(load("patch_runtime").enable_sm120_v41_indexer(source), namespace)
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model))
    )
    if allowed:
        assert namespace["guard"](config)
    else:
        with pytest.raises(ValueError, match="unsupported"):
            namespace["guard"](config)


def test_default_high_is_set_at_l1_and_ui():
    compose = yaml.safe_load((EXAMPLE / "compose.yaml").read_text())
    spec = json.loads((EXAMPLE / "example.json").read_text())
    worker = compose["services"]["deepseek-0"]
    command = worker["command"]
    defaults = json.loads(command[command.index("--default-chat-template-kwargs") + 1])
    assert defaults == {"thinking": True, "reasoning_effort": "high"}
    assert worker["environment"]["VLLM_USE_RUST_FRONTEND"] == "0"
    ui = spec["workspace"]["default_params"]
    assert ui["reasoning_effort"] == spec["model"]["default_reasoning_effort"] == "high"
    assert spec["model"]["reasoning_effort_budgets"]["high"] == 75


def test_evidence_rejects_stale_pcie_candidate_with_identical_command(monkeypatch):
    verification = load("verification")
    compose = yaml.safe_load((EXAMPLE / "compose.yaml").read_text())
    worker = compose["services"]["deepseek-0"]
    environment = dict(worker["environment"])
    container = {
        "Name": "test-worker",
        "Image": verification.SPEC["vllm"]["image_id"],
        "Config": {"Cmd": worker["command"]},
        "State": {"StartedAt": "test-start"},
        "HostConfig": {"DeviceRequests": []},
    }

    def inspect(command, **kwargs):
        if command[1] == "inspect":
            container["Config"]["Env"] = [f"{key}={value}" for key, value in environment.items()]
            return json.dumps([container])
        return (EXAMPLE / "kairyu.yaml").read_bytes()

    monkeypatch.setattr(verification.subprocess, "check_output", inspect)
    monkeypatch.setattr(verification.WORKSPACE_GATES, "runtime_evidence", lambda: {})
    assert verification.runtime_evidence()["replicas"][0]["started_at"] == "test-start"
    flag = "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC"
    environment[flag] = "0" if environment.get(flag, "0") == "1" else "1"
    with pytest.raises(ValueError, match="PCIe IPC"):
        verification.runtime_evidence()


def test_readiness_rejects_absent_tool_calls(monkeypatch):
    control = load("control")
    sent = []

    def post(url, payload, **kw):
        sent.append((url, payload))
        return {"status": "completed", "output": [], "usage": {}}

    monkeypatch.setattr(control, "_post_json", post)
    with pytest.raises(SystemExit, match="did not return an executable"):
        control._validate_tool_calling("http://unused")
    ((url, payload),) = sent
    assert url == "http://unused/v1/responses"
    assert payload["tools"][0]["name"] == "bash" and payload["store"] is False


@pytest.mark.parametrize(
    "answer,status", [("blue", "completed"), ("red", "incomplete"), ("", "completed")]
)
def test_readiness_image_requires_correct_completed_content(monkeypatch, answer, status):
    control = load("control")
    message = {"type": "message", "content": [{"type": "output_text", "text": answer}]}
    monkeypatch.setattr(
        control,
        "_post_json",
        lambda *a, **kw: {
            "status": status,
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [message],
        },
    )
    with pytest.raises(SystemExit, match="image probe"):
        control._validate_vision("http://unused")


@pytest.mark.asyncio
async def test_benchmark_records_reasoning_ttft_without_inventing_visible_content():
    benchmark = load("benchmark")

    async def lines():
        for event in (
            {"type": "response.reasoning_text.delta", "delta": "Let me compute."},
            {
                "type": "response.incomplete",
                "response": {
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "usage": {"input_tokens": 8, "output_tokens": 4},
                },
            },
        ):
            yield f"event: {event['type']}"
            yield "data: " + json.dumps(event)

    result = await benchmark.collect(lines(), time.perf_counter())
    assert result["model_ttft_ms"] >= 0
    assert result["content_ttft_ms"] is None
    assert result["content"] == ""
    assert result["finish_reason"] == "length"
    assert result["completion_tokens"] == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["terminal", "usage", "output"])
async def test_benchmark_rejects_partial_streams(missing):
    benchmark = load("benchmark")

    async def lines():
        if missing != "output":
            yield 'data: {"type":"response.output_text.delta","delta":"323"}'
        if missing != "terminal":
            response = {"status": "completed"}
            if missing != "usage":
                response["usage"] = {"output_tokens": 1}
            yield "data: " + json.dumps({"type": "response.completed", "response": response})

    with pytest.raises(ValueError, match="Incomplete"):
        await benchmark.collect(lines(), time.perf_counter())
