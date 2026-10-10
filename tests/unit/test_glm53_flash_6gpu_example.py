"""Contracts of the six-GPU GLM-5.3-Flash example (one DP6 / EP6 replica)."""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from kairyu.deploy.spec import load_deployment_spec
from kairyu.engine.config_validation import validate_backend_options

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/glm-5.3-flash-6gpu"


@pytest.fixture
def example(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE))
    saved = {name: sys.modules.pop(name, None) for name in ("control", "verification", "tune")}
    yield importlib.import_module
    for name, module in saved.items():
        sys.modules.pop(name, None)
        if module is not None:
            sys.modules[name] = module


def _command() -> list[str]:
    return yaml.safe_load((EXAMPLE / "compose.yaml").read_text())["services"]["glm"]["command"]


def test_served_configuration_loads_and_agrees(example):
    """The gateway must start on the committed YAML and describe the same L1."""
    spec = json.loads((EXAMPLE / "example.json").read_text())
    command = _command()
    deployment = load_deployment_spec((EXAMPLE / "kairyu.yaml").read_text())
    (replica,) = deployment.pools[spec["allocation"]["model"]].replicas
    validate_backend_options(replica.backend, replica.options)

    def flag(name):
        return command[command.index(name) + 1]

    allocation = spec["allocation"]
    assert (
        int(flag("--tensor-parallel-size"))
        == replica.options["tensor_parallel_size"]
        == allocation["tensor_parallel_size"]
    )
    assert (
        int(flag("--data-parallel-size"))
        == replica.options["attention_data_parallel_size"]
        == allocation["data_parallel_size"]
    )
    assert replica.options["expert_parallel_size"] == allocation["expert_parallel_size"]
    assert "--enable-expert-parallel" in command
    assert int(flag("--max-model-len")) == replica.options["max_model_len"]
    # Kairyu admits exactly what the DP engines run at once.
    assert deployment.server.max_concurrency == allocation["data_parallel_size"] * int(
        flag("--max-num-seqs")
    )
    assert replica.options["model_revision"] == spec["model"]["revision"]
    assert spec["vllm"]["repo_digest"].endswith(replica.options["container_image_digest"])
    assert ("--speculative-config" in command) == replica.options["mtp_enabled"]
    example("control")  # import-time allocation check: TP x DP tiles the GPUs


def test_probe_gate_requires_every_dp_engine(example):
    verification = example("verification")
    before = verification.engine_success_counts(
        'vllm:request_success_total{engine="0",finished_reason="stop"} 4\n'
        'vllm:request_success_total{engine="1",finished_reason="stop"} 4\n'
    )
    after = verification.engine_success_counts(
        'vllm:request_success_total{engine="0",finished_reason="stop"} 9\n'
        'vllm:request_success_total{engine="0",finished_reason="length"} 1\n'
        'vllm:request_success_total{engine="1",finished_reason="stop"} 4\n'
        'vllm:request_success_total{engine="2",finished_reason="stop"} 2\n'
    )
    assert verification.served_engines(before, after) == {"0", "2"}


def test_a_tp_group_must_share_a_numa_node(example):
    control = example("control")
    pairs = [[0, 1], [2, 3], [4, 5]]
    assert control.dp_rank_numa_layout({0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}, pairs) == [0, 1, 2]
    with pytest.raises(SystemExit, match="DP rank 1"):
        control.dp_rank_numa_layout({0: 0, 1: 0, 2: 1, 3: 2, 4: 2, 5: 2}, pairs)


def test_chat_ui_switch_is_one_the_gateway_forwards():
    """The Chat UI sends clear_thinking; Kairyu rejects template kwargs it does not allow."""
    spec = importlib.util.spec_from_file_location(
        "glm_effort_filter", EXAMPLE / "webui-reasoning-effort-filter.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    deployment = load_deployment_spec((EXAMPLE / "kairyu.yaml").read_text())
    (replica,) = deployment.pools["glm-5.3-flash"].replicas
    allowed = set(replica.options["capabilities"]["allow_chat_template_kwargs"])

    flt = module.Filter()

    def user(effort):
        return {"valves": flt.UserValves(reasoning_effort=effort)}

    body = flt.inlet({"reasoning_effort": "low", "chat_template_kwargs": {"x": 1}}, user("default"))
    assert "reasoning_effort" not in body
    assert set(body["chat_template_kwargs"]) <= allowed
    assert body["chat_template_kwargs"] == {"clear_thinking": True}
    assert flt.inlet({}, user("high"))["reasoning_effort"] == "high"


@pytest.mark.parametrize(
    "logprob,content,finish,error",
    [
        (-0.01, "323", "stop", None),
        (float("nan"), "323", "stop", "non-finite"),
        (-0.01, "324", "stop", "expected '323'"),
        (-0.01, "323", "length", "finish_reason"),
    ],
)
def test_readiness_probe_rejects_wrong_or_non_finite_answers(
    example, logprob, content, finish, error
):
    control = example("control")
    body = {
        "choices": [
            {
                "finish_reason": finish,
                "message": {"content": content},
                "logprobs": {"content": [{"token": "3", "logprob": logprob}]},
            }
        ]
    }
    result = control.arithmetic_answer_error(body)
    assert (result is None) if error is None else (error in result)


@pytest.mark.parametrize(
    "variant,prompt,ok",
    [
        (
            "default",
            "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>q<|assistant|><think>",
            True,
        ),
        ("low", "[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>q<|assistant|><think>", True),
        ("low", "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>q<|assistant|><think>", False),
        ("high", "[gMASK]<sop><|system|>Reasoning Effort: High<|user|>q<|assistant|>", False),
    ],
)
def test_rendered_effort_follows_the_official_template(example, variant, prompt, ok):
    assert (example("verification").rendering_error(variant, prompt) is None) is ok


@pytest.mark.parametrize("tokens,ok", [(["1,061,738"], True), (["1,048,575"], False), ([], False)])
def test_fit_rule_needs_one_full_context_in_the_kv_pool(example, tokens, ok):
    assert (example("verification").kv_pool_error({"kv_cache_tokens": tokens}) is None) is ok


def test_tuning_candidates_edit_only_their_flags(example):
    tune = example("tune")
    base = _command()
    assert tune.candidate_command(base, "baseline") == base
    no_mtp = tune.candidate_command(base, "no-mtp")
    assert "--speculative-config" not in no_mtp and len(no_mtp) == len(base) - 2
    dp6 = tune.candidate_command(base, "dp6")
    assert dp6[dp6.index("--data-parallel-size") + 1] == "6"
    assert "--speculative-config" not in dp6 and "--disable-custom-all-reduce" not in dp6
    batch = tune.candidate_command(base, "batch-4k")
    assert batch[batch.index("--max-num-batched-tokens") + 1] == "4096"
    # A candidate that collapses onto the committed command measures nothing.
    changed = {
        name: tuple(tune.candidate_command(base, name))
        for name in tune.CANDIDATES
        if name not in {"baseline", "baseline-repeat"}
    }
    assert len(set(changed.values())) == len(changed) and tuple(base) not in changed.values()


def test_api_check_follows_the_bind_address(example):
    control = example("control")
    assert control.api_check_url({"API_PORT": "8015"}) == "http://127.0.0.1:8015"
    assert (
        control.api_check_url({"API_BIND_ADDRESS": "192.0.2.10", "API_PORT": "8015"})
        == "http://192.0.2.10:8015"
    )


_CALL = {"index": 0, "function": {"name": "bash", "arguments": '{"command": "ls"}'}}


@pytest.mark.parametrize(
    "events,error",
    [
        ([{"tool_calls": [_CALL]}, "tool_calls", "[DONE]"], None),
        ([{"tool_calls": [_CALL]}, "tool_calls", {"error": {"message": "x"}}], "stream error"),
        ([{"tool_calls": [_CALL]}, "tool_calls"], "terminal markers"),
    ],
)
def test_streamed_tool_call_gate_rejects_broken_streams(example, events, error):
    lines = []
    for event in events:
        if event == "[DONE]":
            body = "[DONE]"
        elif event == "tool_calls":
            body = json.dumps({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
        elif "error" in event:
            body = json.dumps(event)
        else:
            body = json.dumps({"choices": [{"delta": event}]})
        lines.append(f"data: {body}")
    result = example("verification").streamed_tool_call_error("\n".join(lines))
    assert (result is None) if error is None else (error in result)
