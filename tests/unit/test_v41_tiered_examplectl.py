"""CPU contracts for examples/qwen3.8-deepseek-v4.1-8gpu.

The served YAML is loaded through the real deployment and DSL loaders and
cross-checked against example.json (what the launcher's readiness gate and
the verification compare against the live /routing report); the shipped
DAG runs end to end on the real Conductor with scripted engines; the
verification's row validator and gate arithmetic run on synthetic rows.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import re
from dataclasses import fields
from pathlib import Path

import pytest
import yaml

from kairyu.deploy.spec import load_deployment_spec
from kairyu.dsl import loader as dsl_loader
from kairyu.dsl.loader import load_spec
from kairyu.engine.backend import GenerationResult
from kairyu.engine.prompt import MultimodalItem, MultimodalPrompt, prompt_text
from kairyu.orchestration.budget import Budget
from kairyu.orchestration.conductor import Conductor, RoleSpec
from kairyu.outputs import CompletionOutput
from kairyu.sampling_params import SamplingParams

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/qwen3.8-deepseek-v4.1-8gpu"
SPEC = json.loads((EXAMPLE / "example.json").read_text())
ORCH = SPEC["orchestration"]
TEMPLATE = "l1-qwen3.8-27b-vllm-chat-template.jinja"
ROLE_NAMES = tuple(ORCH["roles"])


def _load(path: Path, name: str):
    module_spec = importlib.util.spec_from_file_location(name, path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_v41_tiered_compose_places_deepseek_on_six_gpus_and_qwen_on_two() -> None:
    services = yaml.safe_load((EXAMPLE / "compose.yaml").read_text())["services"]
    assert set(services) == {"qwen-0", "qwen-1", "deepseek", "kairyu", "chat-ui"}

    def devices(name: str) -> list[str]:
        return services[name]["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"]

    assert devices("qwen-0") == ["6"] and devices("qwen-1") == ["7"]
    assert devices("deepseek") == [str(gpu) for gpu in SPEC["allocation"]["tier2"]["gpu_ids"]]
    tier2 = SPEC["allocation"]["tier2"]
    command = services["deepseek"]["command"]
    assert _option(command, "--tensor-parallel-size") == str(tier2["tensor_parallel_size"])
    assert _option(command, "--data-parallel-size") == str(tier2["attention_data_parallel_size"])
    assert "--enable-expert-parallel" in command
    assert "--speculative-config" not in command
    assert "--default-chat-template-kwargs" not in command
    assert _option(command, "--gpu-memory-utilization") == str(tier2["gpu_memory_utilization"])
    assert _option(command, "--max-num-batched-tokens") == str(tier2["max_num_batched_tokens"])
    assert _option(command, "--max-num-seqs") == str(tier2["max_num_seqs"])
    assert json.loads(_option(command, "--engram-config")) == {
        "cpu_offload": tier2["engram_cpu_offload"]
    }
    assert json.loads(_option(command, "--limit-mm-per-prompt")) == {"image": 1}
    for flag in ("--tokenizer-mode", "--reasoning-parser", "--tool-call-parser"):
        assert _option(command, flag) == "deepseek_v41"
    assert _option(command, "--max-model-len") == str(SPEC["models"]["tier2"]["max_context_tokens"])
    assert services["deepseek"]["environment"]["VLLM_USE_RUST_FRONTEND"] == "0"
    assert SPEC["vllm"]["deepseek"]["image"] in services["deepseek"]["image"]
    assert services["deepseek"]["ports"] == ["127.0.0.1:${DEEPSEEK_L1_PORT:-8009}:8000"]

    qwen_command = services["qwen-0"]["command"]
    assert qwen_command == services["qwen-1"]["command"]
    assert _option(qwen_command, "--chat-template") == f"/etc/kairyu/{TEMPLATE}"
    assert _option(qwen_command, "--max-model-len") == "262144"
    assert _option(qwen_command, "--reasoning-parser") == "qwen3"
    assert _option(qwen_command, "--tool-call-parser") == "qwen3_coder"
    assert "--default-chat-template-kwargs" not in qwen_command
    for name in ("qwen-0", "qwen-1"):
        assert f"./{TEMPLATE}:/etc/kairyu/{TEMPLATE}:ro" in services[name]["volumes"]

    volumes = services["kairyu"]["volumes"]
    for mount in (
        "./kairyu.yaml:/etc/kairyu/kairyu.yaml:ro",
        "./auto-max.yaml:/etc/kairyu/auto-max.yaml:ro",
        "./ensemble-max.yaml:/etc/kairyu/ensemble-max.yaml:ro",
        "./router.json:/etc/kairyu/router.json:ro",
    ):
        assert mount in volumes
    assert any(
        isinstance(volume, dict) and volume.get("target") == "/var/lib/kairyu/placement"
        for volume in volumes
    )
    assert services["kairyu"]["ports"] == ["${API_BIND_ADDRESS:-0.0.0.0}:${API_PORT:-8008}:8000"]
    assert set(services["kairyu"]["depends_on"]) == {"qwen-0", "qwen-1", "deepseek"}

    ui = services["chat-ui"]["environment"]
    assert ui["OPENAI_API_BASE_URL"] == "http://kairyu:8000/v1"
    assert json.loads(ui["OPENAI_API_CONFIGS"]) == {"0": {"model_ids": ["kairyu-auto-max"]}}
    assert json.loads(ui["DEFAULT_MODEL_PARAMS"]) == {"max_tokens": 65536, "stream_response": False}
    assert services["chat-ui"]["depends_on"] == {"kairyu": {"condition": "service_healthy"}}
    assert services["chat-ui"]["ports"] == [
        "${CHAT_UI_BIND_ADDRESS:-0.0.0.0}:${CHAT_UI_PORT:-3008}:8080"
    ]


def test_v41_tiered_gateway_binds_two_pools_and_two_orchestrators() -> None:
    deployment = load_deployment_spec(
        (EXAMPLE / "kairyu.yaml").read_text(), resolve_credentials=False
    )
    assert deployment.server.max_concurrency == 256
    qwen = deployment.pools["qwen3.8-27b"]
    assert len(qwen.replicas) == 2
    assert qwen.queue_depth_threshold == 0 and qwen.prefix_index and qwen.placement_log_path
    for replica in qwen.replicas:
        options = replica.options
        assert options["capabilities"]["allow_chat_template_kwargs"] == ["enable_thinking"]
        assert options["capabilities"]["allow_prompt_kinds"] == ["multimodal"]
        assert options["image_input_policy"]["max_images"] == 1
        assert options["image_input_policy"]["max_image_bytes"] == 8388608
        assert options["image_input_policy"]["max_image_pixels"] == 2097152
        assert options["max_model_len"] == 262144
        assert options["container_image_digest"] == SPEC["vllm"]["qwen"]["image_id"]
    deepseek = deployment.pools["deepseek-v4.1-flash"]
    assert len(deepseek.replicas) == 1
    options = deepseek.replicas[0].options
    tier2 = SPEC["allocation"]["tier2"]
    assert options["tensor_parallel_size"] == tier2["tensor_parallel_size"]
    assert options["expert_parallel_size"] == tier2["expert_parallel_size"]
    assert options["attention_data_parallel_size"] == tier2["attention_data_parallel_size"]
    assert options["dspark_enabled"] is False
    assert options["container_image_digest"] == SPEC["vllm"]["deepseek"]["image_id"]
    assert options["model_revision"] == SPEC["models"]["tier2"]["revision"]
    assert options["max_model_len"] == 1048576
    assert options["capabilities"]["allow_chat_template_kwargs"] == ["enable_thinking"]
    assert options["capabilities"]["allow_prompt_kinds"] == ["multimodal"]
    assert options["image_input_policy"]["max_images"] == 1
    assert options["image_input_policy"]["max_image_pixels"] == 2097152
    assert deepseek.placement_log_path
    assert {name: section.spec for name, section in deployment.orchestrators.items()} == {
        "kairyu-auto-max": "/etc/kairyu/auto-max.yaml",
        "kairyu-ensemble-max": "/etc/kairyu/ensemble-max.yaml",
    }
    assert deployment.public_models == frozenset(
        {"kairyu-auto-max", "kairyu-ensemble-max", "embed-small"}
    )
    assert deployment.legacy_chat_models == frozenset({"qwen3.8-27b", "deepseek-v4.1-flash"})
    assert deployment.chat_templates == {}
    assert "embed-small" in deployment.embeddings


def test_v41_tiered_l2_pins_the_deepseek_led_dag_and_its_forced_twin() -> None:
    spec = load_spec(EXAMPLE / "auto-max.yaml")
    roles = {role.name: role for role in spec.roles}
    assert list(roles) == list(ROLE_NAMES)
    assert {worker.name: worker.engine_ref for worker in spec.workers} == {
        "tier1": "qwen3.8-27b",
        "tier2": "deepseek-v4.1-flash",
    }
    assert spec.router.kind == "calibrated" and spec.router.target_mode == "auto-max"
    assert spec.router.sha256 == hashlib.sha256((EXAMPLE / "router.json").read_bytes()).hexdigest()
    for name, role in roles.items():
        assert role.worker == ORCH["role_workers"][name]
        assert role.reasoning_effort == ORCH["role_efforts"][name]
        for field_name in ("prompt", "prompt_headless"):
            text = getattr(role, field_name)
            if text:
                # The L3-rendered conversation enters every prompt exactly once.
                assert text.count("{query}") == 1, (name, field_name)
    head = roles["head"]
    assert head.role_type == "head" and head.depends_on == () and head.sampling.max_tokens == 256
    requirements = roles["requirements"]
    assert requirements.depends_on == ()
    assert requirements.sampling.max_tokens_by_effort.model_dump() == {
        "low": 16384,
        "high": 32768,
        "max": 65536,
    }
    independent = roles["independent"]
    assert independent.depends_on == ()
    assert "{requirements}" not in independent.prompt and "{policies}" not in independent.prompt
    assert roles["policies"].depends_on == ("requirements",)
    for index in range(1, 5):
        answer = roles[f"answer_{index}"]
        assert answer.depends_on == ("policies", "requirements")
        assert f"POLICY {index}" in answer.prompt
        assert answer.sampling.seed_offset == index
        assert answer.sampling.max_tokens == 16384 and answer.sampling.top_k == 20
    candidates = {"independent", "answer_1", "answer_2", "answer_3", "answer_4"}
    synthesis = roles["synthesis"]
    assert set(synthesis.depends_on) == candidates | {"requirements"}
    assert "=== DECISION RECORD ===" in synthesis.prompt
    final = roles["final"]
    assert final.role_type == "publisher"
    assert set(final.depends_on) == candidates | {"head", "synthesis", "requirements"}
    assert final.sampling.max_tokens is None and final.sampling.max_tokens_by_effort is None
    assert (
        "{head}" in final.prompt and final.prompt_headless and "{head}" not in final.prompt_headless
    )
    audit = roles["audit"]
    assert audit.role_type == "verifier" and audit.verifies == "final"
    assert set(audit.depends_on) == {"final", "head", "requirements"}
    assert spec.budget.max_steps == ORCH["max_steps"] == 19
    assert spec.budget.max_refine_depth == ORCH["product_max_refinements"] == 2
    assert spec.internal_max_tokens == ORCH["internal_max_output_tokens"] == 131072
    assert spec.public_output_floor == 256
    assert spec.default_reasoning_effort == ORCH["default_reasoning_effort"] == "high"
    assert spec.expose_intermediate_outputs is True and spec.moa_samples == 0

    profiles = {profile.name: profile.roles for profile in spec.profiles}
    assert {name: [role.name for role in rs] for name, rs in profiles.items()} == ORCH["profiles"]
    for name, expected in ORCH["direct_route_sampling"].items():
        sampling = {
            key: value
            for key, value in profiles[name][0].sampling.model_dump().items()
            if value not in (None, [], {}, ())
        }
        assert sampling == expected, name
    qwen_answer = profiles["qwen_direct"][0]
    assert qwen_answer.reasoning_effort is None and qwen_answer.sampling.max_tokens is None
    qwen_think = profiles["qwen_think_medium"][0]
    assert qwen_think.reasoning_effort == "high" and qwen_think.sampling.max_tokens is None
    assert qwen_think.reasoning_continuation == "chat" and qwen_think.reasoning_close_tag
    deepseek_answer = profiles["deepseek_direct"][0]
    assert deepseek_answer.worker == "tier2" and deepseek_answer.reasoning_effort is None
    assert deepseek_answer.prompt_suffix == "" and not deepseek_answer.reasoning_closed
    assert profiles["deepseek_think"][0].reasoning_effort == "inherit"
    judge = spec.profile_judge
    assert judge.worker == "tier1" and judge.fallback == "primary"
    assert [{"label": choice.label, "profile": choice.profile} for choice in judge.choices] == ORCH[
        "profile_judge"
    ]["choices"]
    assert {ROLE_NAMES[-2]: "final"} == {"final": ORCH["profile_final_roles"]["primary"]}

    ensemble = load_spec(EXAMPLE / "ensemble-max.yaml")
    assert ensemble.roles == spec.roles
    assert ensemble.workers == spec.workers and ensemble.router == spec.router
    assert ensemble.budget == spec.budget
    assert ensemble.profiles == () and ensemble.profile_judge is None
    for field_name in (
        "internal_max_tokens",
        "public_output_floor",
        "default_reasoning_effort",
        "expose_intermediate_outputs",
        "moa_samples",
        "shared_prefix",
    ):
        assert getattr(ensemble, field_name) == getattr(spec, field_name), field_name


# --- the shipped DAG on the real Conductor -----------------------------------

CHECKLIST = json.dumps(
    [
        {
            "id": "R1",
            "priority": "minimum",
            "requirement": "Compare both options",
            "acceptance_criterion": "Discuss A and B",
            "source": "Compare A and B",
        }
    ]
)
POLICIES = "POLICY 1: cost.\nPOLICY 2: speed.\nPOLICY 3: risk.\nPOLICY 4: fit."
PROPOSAL = "A is inexpensive; B is faster.\n=== DECISION RECORD ===\nAdopted candidate 5."
DEFECT = "FAIL\nR1 | unsatisfied | evidence: B is absent | correction: add B"
_ROLE_MARKER = re.compile(r"\[(" + "|".join(ROLE_NAMES) + r")\]")


class ScriptedBackend:
    """Answers each role by name and records exactly what it was sent."""

    def __init__(self, *, audit_outputs: list[str]):
        self.audit_outputs = list(audit_outputs)
        self.requests: dict[str, list] = {}

    def supports_prompt_kind(self, kind: str) -> bool:
        return kind in {"text", "multimodal"}

    def supports_chat_template_kwargs(self, keys) -> bool:
        return set(keys) <= {"enable_thinking"}

    async def generate(self, request):
        prompt = request.prompt
        text = prompt_text(prompt.base if isinstance(prompt, MultimodalPrompt) else prompt)
        role = _ROLE_MARKER.findall(text)[-1]
        self.requests.setdefault(role, []).append((request, text))
        if role == "head":
            output = "Comparison:"
        elif role == "requirements":
            output = CHECKLIST
        elif role == "policies":
            output = POLICIES
        elif role == "synthesis":
            output = PROPOSAL
        elif role == "final":
            attempt = len(self.requests[role])
            output = (
                "\nA is inexpensive."
                if attempt == 1
                else f"\nA is inexpensive. B is faster (attempt {attempt})."
            )
        elif role == "audit":
            output = self.audit_outputs[
                min(len(self.requests[role]) - 1, len(self.audit_outputs) - 1)
            ]
        else:
            output = f"Candidate text from {role}."
        return GenerationResult(
            request_id=request.request_id,
            prompt=prompt,
            completions=(CompletionOutput(index=0, text=output, token_ids=()),),
        )

    async def stream(self, request):
        yield await self.generate(request)

    async def shutdown(self) -> None:
        pass


def _conductor_roles(spec) -> tuple[RoleSpec, ...]:
    roles = []
    for role in spec.roles:
        values = {field.name: getattr(role, field.name) for field in fields(RoleSpec)}
        values["sampling"] = dsl_loader._role_sampling(role)
        values["executor"] = None
        roles.append(RoleSpec(**values))
    return tuple(roles)


def _run_dag(*, with_image: bool, head_enabled: bool, audit_outputs: list[str], effort="max"):
    spec = load_spec(EXAMPLE / "auto-max.yaml")
    backend = ScriptedBackend(audit_outputs=audit_outputs)
    media = (
        MultimodalPrompt(
            "Compare A and B",
            (MultimodalItem("image", "uri", "data:image/png;base64,AAAA"),),
        )
        if with_image
        else None
    )
    conductor = Conductor(
        roles=_conductor_roles(spec),
        workers={worker.name: backend for worker in spec.workers},
        sampling_params=SamplingParams(max_tokens=65536),
        final_structured_format_in_prompt=not head_enabled,
        multimodal_prompt=media,
        reasoning_effort=effort,
        expose_intermediate_outputs=True,
    )
    result = asyncio.run(
        asyncio.wait_for(
            conductor.run(
                "Compare A and B",
                Budget(
                    max_steps=spec.budget.max_steps, max_refine_depth=spec.budget.max_refine_depth
                ),
            ),
            timeout=10,
        )
    )
    return backend, result


@pytest.mark.parametrize("with_image", [False, True])
@pytest.mark.parametrize("head_enabled", [False, True])
@pytest.mark.parametrize("fail_audits", [0, 1, 3])
def test_v41_tiered_dag_runs_requirements_candidates_synthesis_final_audit(
    with_image: bool, head_enabled: bool, fail_audits: int
) -> None:
    audit_outputs = [DEFECT] * fail_audits + [
        "PASS\nR1 | satisfied | evidence: A and B | correction: none"
    ]
    backend, result = _run_dag(
        with_image=with_image, head_enabled=head_enabled, audit_outputs=audit_outputs
    )
    assert result.final_unit_ok, [(event.node, event.detail) for event in result.trace]
    sent = backend.requests
    assert ("head" in sent) is head_enabled
    for name in ROLE_NAMES:
        if name != "head":
            assert name in sent, name

    # Requirements and the independent candidate read only the conversation.
    for name in ("requirements", "independent"):
        _, text = sent[name][0]
        assert "Compare A and B" in text
        assert CHECKLIST not in text and POLICIES not in text and "Candidate text" not in text
    _, policies_text = sent["policies"][0]
    assert CHECKLIST in policies_text and "Candidate text" not in policies_text
    for index in range(1, 5):
        _, text = sent[f"answer_{index}"][0]
        assert CHECKLIST in text and POLICIES in text and f"POLICY {index}" in text
    _, synthesis_text = sent["synthesis"][0]
    for name in ("independent", "answer_1", "answer_2", "answer_3", "answer_4"):
        assert f"Candidate text from {name}." in synthesis_text
    assert CHECKLIST in synthesis_text
    _, final_text = sent["final"][0]
    assert PROPOSAL in final_text and CHECKLIST in final_text
    assert ("Comparison:" in final_text) is head_enabled
    assert ("COMMITTED OPENING" in final_text) is head_enabled
    _, audit_text = sent["audit"][0]
    assert "A is inexpensive." in audit_text and CHECKLIST in audit_text
    assert ("Comparison:" in audit_text) is head_enabled

    # Effort propagation: DeepSeek roles inherit the caller's max; the Qwen
    # answerers stay at their fixed declaration; the head is non-thinking.
    for name in ("requirements", "independent", "policies", "synthesis", "final", "audit"):
        assert sent[name][0][0].reasoning_effort == "max", name
    for index in range(1, 5):
        assert sent[f"answer_{index}"][0][0].reasoning_effort == "high"
    if head_enabled:
        head_request = sent["head"][0][0]
        assert head_request.reasoning_effort is None
        assert head_request.chat_template_kwargs == {"enable_thinking": False}

    # Images reach every role as the original media, DeepSeek roles included.
    for name in sent:
        request = sent[name][0][0]
        assert isinstance(request.prompt, MultimodalPrompt) is with_image, name
        if with_image:
            assert request.prompt.items[0].data == "data:image/png;base64,AAAA"

    # Audit loop: FAIL -> refine (feedback appended) -> re-audit; at most two
    # refinements, then the last attempt is published with the verdict kept.
    attempts = min(fail_audits + 1, 3)
    assert len(sent["audit"]) == attempts and len(sent["final"]) == attempts
    if fail_audits:
        assert DEFECT in sent["final"][1][1]
    verdicts = [
        event for event in result.trace if event.node == "audit" and "pass" in event.metadata
    ]
    assert len(verdicts) == attempts
    assert verdicts[-1].metadata["pass"] is (fail_audits < 3)
    assert verdicts[-1].metadata["refinement_exhausted"] is (fail_audits == 3)
    assert result.outputs["audit"].startswith("FAIL" if fail_audits == 3 else "PASS")
    expected_final = (
        "A is inexpensive." if fail_audits == 0 else f"B is faster (attempt {attempts})"
    )
    assert expected_final in result.final_text
    assert CHECKLIST not in result.final_text and "DECISION RECORD" not in result.final_text
    assert not [event for event in result.trace if event.status == "failed"]


def test_v41_tiered_inconclusive_verdict_reverifies_without_refining() -> None:
    backend, result = _run_dag(
        with_image=False,
        head_enabled=True,
        audit_outputs=[
            "Assessment first, verdict later.",
            "PASS\nR1 | satisfied | evidence: ok | correction: none",
        ],
    )
    assert result.final_unit_ok
    assert len(backend.requests["audit"]) == 2 and len(backend.requests["final"]) == 1
    first = [event for event in result.trace if event.node == "audit"]
    assert [event.metadata.get("inconclusive") for event in first] == [True]
    reverify = [event for event in result.trace if event.node == "audit:reverify"]
    assert [event.metadata.get("pass") for event in reverify] == [True]
    assert reverify[0].metadata["refinement_exhausted"] is False
    assert result.outputs["audit"].startswith("PASS")


# --- verification tooling ----------------------------------------------------------


@pytest.fixture
def verification(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NVME_STORAGE_ROOT", "/mnt/nvme/kairyu")
    return _load(EXAMPLE / "verification.py", "v41_tiered_verification")


def _event(node, kind="generation", status="success", role="proposal", tokens=100, attempt=0):
    return {
        "node": node,
        "kind": kind,
        "status": status,
        "role": role,
        "attempt": attempt,
        "timing": {
            "queued_at": "2026-09-14T00:00:00.000Z",
            "started_at": "2026-09-14T00:00:01.000Z",
            "completed_at": "2026-09-14T00:00:03.000Z",
        },
        "usage": {"prompt_tokens": 10, "completion_tokens": tokens, "cached_tokens": 0},
        "detail": {},
    }


def _primary_sample(*, judged: bool = True) -> dict:
    events = [_event("profile_judge", kind="classification", role="judge")] if judged else []
    events.append(_event("head", role="head", tokens=40))
    for node in ROLE_NAMES:
        if node in {"head", "audit", "final"}:
            continue
        events.append(_event(node, role="planner"))
    events.append(_event("final", role="publisher", tokens=800))
    events.append(_event("audit", kind="verification", role="verifier"))
    return {
        "index": 0,
        "passed": True,
        "content": "answer",
        "content_ttft_ms": 900.0,
        "total_ms": 20000.0,
        "completion_tokens": 840,
        "trace": {"events": events},
    }


def test_v41_tiered_row_validator_requires_every_role_and_flags_cut_offs(verification) -> None:
    caps = verification.role_caps("high", 65536)
    assert caps["requirements"] == 32768 and caps["synthesis"] == 65536
    assert caps["answer_1"] == 16384 and caps["audit"] == 16384 and caps["head"] == 256
    assert "final" not in caps
    good = _primary_sample()
    assert verification.sample_problems(good, judged=True, require_head=True, caps=caps) == []

    missing_audit = _primary_sample()
    missing_audit["trace"]["events"] = [
        e for e in missing_audit["trace"]["events"] if e["node"] != "audit"
    ]
    problems = verification.sample_problems(
        missing_audit, judged=True, require_head=True, caps=caps
    )
    assert any("audit" in problem for problem in problems)

    failed_candidate = _primary_sample()
    failed_candidate["trace"]["events"].append(
        {**_event("answer_2", status="failed"), "error": {"type": "UpstreamClientError"}}
    )
    problems = verification.sample_problems(
        failed_candidate, judged=True, require_head=True, caps=caps
    )
    assert any("failed stages: answer_2" in problem for problem in problems)

    cut_off = _primary_sample()
    for event in cut_off["trace"]["events"]:
        if event["node"] == "answer_1":
            event["usage"]["completion_tokens"] = 16384
    problems = verification.sample_problems(cut_off, judged=True, require_head=True, caps=caps)
    assert any("answer_1 ended at its 16384-token cap" in problem for problem in problems)

    headless = _primary_sample()
    headless["trace"]["events"] = [
        e for e in headless["trace"]["events"] if e["node"] != "head"
    ] + [
        {
            **_event("head", role="head", status="skipped", tokens=0),
            "detail": {"reason": "intent", "head": True},
        }
    ]
    assert verification.sample_problems(headless, judged=True, require_head=True, caps=caps) == []
    no_head = _primary_sample()
    no_head["trace"]["events"] = [e for e in no_head["trace"]["events"] if e["node"] != "head"]
    assert any(
        "head" in p
        for p in verification.sample_problems(no_head, judged=True, require_head=True, caps=caps)
    )

    unjudged = _primary_sample(judged=False)
    assert verification.sample_problems(unjudged, judged=True, require_head=True, caps=caps)
    assert verification.sample_problems(unjudged, judged=False, require_head=True, caps=caps) == []

    direct = {
        **_primary_sample(),
        "trace": {
            "events": [
                _event("profile_judge", kind="classification", role="judge"),
                _event("qwen_answer", role="publisher"),
            ]
        },
    }
    assert verification.sample_problems(direct, judged=True, require_head=True, caps=caps) == []
    assert verification.sample_problems(direct, judged=False, require_head=True, caps=caps)

    routes = verification._route_report([good, direct])
    assert routes["routes"]["primary"]["requests"] == 1
    assert routes["routes"]["qwen_direct"]["requests"] == 1
    assert routes["judged_samples"] == 2
    assert verification.gated_ttft_p50(routes) == 900.0


def test_v41_tiered_baseline_and_placement_gates(verification) -> None:
    def sample(**overrides):
        return {
            "passed": True,
            "finish_reason": "stop",
            "content": "x",
            "content_ttft_ms": 1000.0,
            **overrides,
        }

    valid = {"samples": [sample(content_ttft_ms=float(i)) for i in range(1, 33)]}
    ttft, source = verification.baseline_ttft_p50(valid, 32)
    assert ttft == 16.0 and source == "paired_direct"
    cut = {"samples": [sample() for _ in range(31)] + [sample(finish_reason="length")]}
    assert verification.baseline_ttft_p50(cut, 32)[0] is None
    empty = {"samples": [sample() for _ in range(31)] + [sample(content="  ")]}
    assert verification.baseline_ttft_p50(empty, 32)[0] is None
    assert verification.baseline_ttft_p50(valid, 16)[0] is None

    from collections import Counter

    even = verification.placement_report(
        Counter({"0": 17, "1": 15}), replicas=2, gated=True, max_share_of_mean=1.25
    )
    assert even["passed"] is True
    skewed = verification.placement_report(
        Counter({"0": 30, "1": 2}), replicas=2, gated=True, max_share_of_mean=1.25
    )
    assert skewed["passed"] is False
    assert (
        verification.placement_report(
            Counter({"0": 32}), replicas=2, gated=False, max_share_of_mean=1.25
        )["passed"]
        is None
    )


def test_v41_tiered_verification_rejects_storage_outside_nvme(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVME_STORAGE_ROOT", "/tmp/not-nvme")
    with pytest.raises(SystemExit, match="/mnt/nvme"):
        _load(EXAMPLE / "verification.py", "v41_tiered_verification_nvme")


# --- launcher ------------------------------------------------------------------------


def test_v41_tiered_control_maps_gpus_and_guards_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    control = _load(EXAMPLE / "control.py", "v41_tiered_control")
    text = "\n".join(
        f"{index}, NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 12.0, "
        f"00000000:{16 + index:02x}:00.0"
        for index in range(8)
    )
    rows = control._gpu_inventory(text)
    assert sorted(rows) == list(range(8))
    env: dict[str, str] = {}
    control._assign_cpusets(env, {index: f"cpus-{index}" for index in range(8)})
    assert env["QWEN_0_CPUSET"] == "cpus-6" and env["QWEN_1_CPUSET"] == "cpus-7"
    assert env["DEEPSEEK_CPUSET"] == ",".join(f"cpus-{index}" for index in range(6))
    monkeypatch.setenv("NVME_STORAGE_ROOT", "/tmp/not-nvme")
    with pytest.raises(SystemExit, match="/mnt/nvme"):
        control._nvme_root()


def _served_policy(*, judged: bool) -> dict:
    spec = load_spec(EXAMPLE / ("auto-max.yaml" if judged else "ensemble-max.yaml"))
    policy = {
        "roles": [{"name": role.name} for role in spec.roles],
        "stream_head": "head",
        "moa_samples": 0,
        "budget": {
            "max_steps": spec.budget.max_steps,
            "max_refine_depth": spec.budget.max_refine_depth,
        },
        "expose_intermediate_outputs": True,
        "configured_engines": {
            "tier1": {"model": "qwen3.8-27b"},
            "tier2": {"model": "deepseek-v4.1-flash"},
        },
    }
    if judged:
        policy["profiles"] = {
            profile.name: [
                {
                    "name": role.name,
                    # The live report is JSON: tuples arrive as lists.
                    "sampling": (
                        json.loads(json.dumps(role.sampling.model_dump()))
                        if role.sampling
                        else None
                    ),
                }
                for role in profile.roles
            ]
            for profile in spec.profiles
        }
        policy["profile_judge"] = {
            "worker": spec.profile_judge.worker,
            "fallback": spec.profile_judge.fallback,
            "choices": [
                {"label": choice.label, "profile": choice.profile}
                for choice in spec.profile_judge.choices
            ],
        }
    return policy


def test_v41_tiered_readiness_gate_pins_both_served_policies() -> None:
    control = _load(EXAMPLE / "control.py", "v41_tiered_control_policy")
    control._validate_policy(_served_policy(judged=True), judged=True)
    control._validate_policy(_served_policy(judged=False), judged=False)
    without_audit = _served_policy(judged=True)
    without_audit["roles"] = [row for row in without_audit["roles"] if row["name"] != "audit"]
    with pytest.raises(SystemExit, match="11-role"):
        control._validate_policy(without_audit, judged=True)
    judged_twin = _served_policy(judged=True)
    with pytest.raises(SystemExit, match="without a judge"):
        control._validate_policy(judged_twin, judged=False)
    wrong_budget = _served_policy(judged=False)
    wrong_budget["budget"]["max_steps"] = 16
    with pytest.raises(SystemExit, match="max_steps"):
        control._validate_policy(wrong_budget, judged=False)
