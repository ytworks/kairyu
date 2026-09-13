from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/qwen3.8-deepseek-v4-8gpu"
EMBEDDING_MODEL_REPOSITORY = "Qdrant/all-MiniLM-L6-v2-onnx"
EMBEDDING_MODEL_REVISION = "5f1b8cd78bc4fb444dd171e59b18f3a3af89a079"
EMBEDDING_MODEL_SHA256 = (
    "bbd7b466f6d58e646fdc2bd5fd67b2f5e93c0b687011bd4548c420f7bd46f0c5"
)
EMBEDDING_PROVENANCE_SHA256 = (
    "57246a4990eb0f08755df06ba57c1fec161032bd588332435e89c7ece244639c"
)


def _load(path: Path, name: str):
    module_spec = importlib.util.spec_from_file_location(name, path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_tiered_browser_gate_requires_one_model_and_separate_reasoning_ui() -> None:
    browser = (ROOT / "scripts/webui_browser_smoke.mjs").read_text()
    wrapper = (EXAMPLE / "browser-smoke.sh").read_text()

    assert "WEBUI_SMOKE_PHASE=tiered" in wrapper
    assert "EXPECTED_PRODUCT_MODEL=kairyu-auto-max" in wrapper
    assert "inventory.body.data.map" in browser
    assert "JSON.stringify([productModel])" in browser
    assert "button[aria-expanded]" in browser
    assert "intermediate processing was not initially folded" in browser
    assert "child !== reasoningRoot" in browser
    assert "child.classList.contains('markdown-prose')" in browser
    for attribution in (
        "L2 role:",
        "L1 worker:",
        "Engine:",
        "Model:",
        "tier1",
        "tier2",
        "qwen3.8-27b",
        "deepseek-v4.1-flash-thinking",
    ):
        assert attribution in browser














def _embedding_response() -> dict:
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": 0, "embedding": [0.0] * 384},
            {"object": "embedding", "index": 1, "embedding": [1.0] * 384},
        ],
        "model": "embed-small",
        "usage": {"prompt_tokens": 6, "total_tokens": 6},
    }


def test_tiered_embedding_smoke_accepts_two_finite_vectors() -> None:
    control = _load(EXAMPLE / "control.py", "tiered_embedding_smoke_valid")

    control._validate_embedding_smoke(_embedding_response())


def test_tiered_readiness_posts_two_input_embedding_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _load(EXAMPLE / "control.py", "tiered_embedding_readiness")
    posts: list[tuple[str, dict]] = []

    def fake_json(url: str) -> dict:
        if url.endswith("/readyz"):
            return {"status": "ready"}
        if url.endswith("/v1/models"):
            return {"data": [{"id": "kairyu-auto-max"}, {"id": "embed-small"}]}
        assert url.endswith("/routing")
        return {
            "models": {
                "kairyu-auto-max": {
                    "roles": [
                        {"name": name}
                        for name in (
                            "head",
                            "requirements",
                            "deepseek_candidate",
                            "policies",
                            "answer_1",
                            "answer_2",
                            "answer_3",
                            "answer_4",
                            "review",
                            "synthesis",
                            "audit",
                        )
                    ],
                    "profiles": {
                        profile: [
                            {
                                "name": roles[0],
                                "sampling": control.SPEC["orchestration"][
                                    "direct_route_sampling"
                                ][profile],
                            }
                        ]
                        for profile, roles in control.SPEC["orchestration"][
                            "profiles"
                        ].items()
                    },
                    "profile_judge": {
                        "worker": "tier1",
                        "fallback": "primary",
                        "choices": [
                            {"label": label, "profile": profile, "criteria": "x"}
                            for label, profile in (
                                ("QWEN", "qwen_direct"),
                                ("QWEN_THINK", "qwen_think_medium"),
                                ("DEEPSEEK", "deepseek_direct"),
                                ("DEEPSEEK_THINK", "deepseek_think"),
                                ("ENSEMBLE", "primary"),
                            )
                        ],
                    },
                    "stream_head": "head",
                    "moa_samples": 0,
                    "budget": {
                        "max_steps": control.SPEC["orchestration"]["max_steps"],
                        "max_steps_per_additional_choice": control.SPEC["orchestration"][
                            "max_steps_per_additional_choice"
                        ],
                        "max_refine_depth": 2,
                    },
                    "expose_intermediate_outputs": True,
                    "configured_engines": {
                        "tier1": {"model": "qwen3.8-27b"},
                        "tier2": {"model": "deepseek-v4.1-flash-thinking"},
                        "tier2-direct": {"model": "deepseek-v4.1-flash"},
                    },
                }
            }
        }

    def fake_post(url: str, payload: dict) -> dict:
        posts.append((url, payload))
        return _embedding_response() if url.endswith("/v1/embeddings") else {"count": 1}

    monkeypatch.setattr(control, "_json_url", fake_json)
    monkeypatch.setattr(control, "_post_json_url", fake_post)

    control._validate_ready("http://api.test", "http://tokenizer.test/tokenize")

    assert posts == [
        (
            "http://api.test/v1/embeddings",
            {
                "model": "embed-small",
                "input": ["kairyu readiness probe", "two-input contract"],
                "encoding_format": "float",
            },
        ),
        (
            "http://tokenizer.test/tokenize",
            {"model": "deepseek-v4.1-flash", "prompt": "kairyu"},
        ),
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("model", "model identity"),
        ("indices", "indices"),
        ("dimensions", "384 dimensions"),
        ("finite", "finite numbers"),
        ("usage", "positive exact usage"),
    ],
)
def test_tiered_embedding_smoke_rejects_malformed_contract(
    mutation: str,
    message: str,
) -> None:
    control = _load(EXAMPLE / "control.py", f"tiered_embedding_smoke_{mutation}")
    response = _embedding_response()
    if mutation == "model":
        response["model"] = "wrong-model"
    elif mutation == "indices":
        response["data"][1]["index"] = 2
    elif mutation == "dimensions":
        response["data"][1]["embedding"].pop()
    elif mutation == "finite":
        response["data"][0]["embedding"][0] = float("inf")
    else:
        response["usage"]["prompt_tokens"] = 0

    with pytest.raises(SystemExit, match=message):
        control._validate_embedding_smoke(response)


class _FakeWebUI:
    """Model the pinned Open WebUI function API, including flip-only toggles."""

    def __init__(self) -> None:
        self.functions: dict[str, dict] = {}
        self.calls: list[str] = []
        self.role = "admin"

    def __call__(self, ui_url, path, *, token=None, payload=None, method=None):
        self.calls.append(path)
        if path == "/api/v1/auths/signin":
            return {"role": self.role, "token": "session-token"}
        assert token == "session-token"
        if path == "/api/v1/functions/":
            return [dict(row) for row in self.functions.values()]
        if path == "/api/v1/functions/create":
            row = {**payload, "is_active": False, "is_global": False}
            self.functions[payload["id"]] = row
            return dict(row)
        prefix = "/api/v1/functions/id/"
        assert path.startswith(prefix)
        function_id, _, action = path[len(prefix) :].partition("/")
        row = self.functions[function_id]
        if action == "update":
            row.update(payload)
            return dict(row)
        if action in {"toggle", "toggle/global"}:
            key = "is_active" if action == "toggle" else "is_global"
            row[key] = not row[key]
            return dict(row)
        assert action == "valves/user/spec"
        return {
            "properties": {
                "reasoning_effort": {"enum": ["default", "low", "high", "max"]}
            }
        }


def test_tiered_chat_ui_effort_filter_controls_request_body() -> None:
    module = _load(
        EXAMPLE / "webui-reasoning-effort-filter.py",
        "tiered_chat_ui_effort_filter",
    )
    selector = module.Filter()

    for effort in ("low", "high", "max"):
        body = {"reasoning_effort": "stale"}
        user = {"valves": selector.UserValves(reasoning_effort=effort)}
        assert selector.inlet(body, user) == {"reasoning_effort": effort}

    body = {"reasoning_effort": "max"}
    user = {"valves": selector.UserValves(reasoning_effort="default")}
    assert selector.inlet(body, user) == {}


def test_tiered_chat_ui_effort_selector_provision_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _load(EXAMPLE / "control.py", "tiered_chat_ui_effort_selector")
    webui = _FakeWebUI()
    monkeypatch.setattr(control, "_webui_api", webui)

    control._provision_chat_ui_effort_selector("http://127.0.0.1:3000")
    row = webui.functions["reasoning_effort"]
    assert row["is_active"] and row["is_global"]
    assert "class Filter" in row["content"]
    assert webui.calls.count("/api/v1/functions/create") == 1
    toggles = [call for call in webui.calls if call.endswith(("toggle", "toggle/global"))]
    assert len(toggles) == 2

    # Re-provisioning must refresh content without flipping activation off.
    webui.calls.clear()
    control._provision_chat_ui_effort_selector("http://127.0.0.1:3000")
    assert webui.functions["reasoning_effort"]["is_active"]
    assert webui.functions["reasoning_effort"]["is_global"]
    assert "/api/v1/functions/create" not in webui.calls
    assert "/api/v1/functions/id/reasoning_effort/update" in webui.calls
    assert not [call for call in webui.calls if call.endswith(("toggle", "toggle/global"))]

    webui.role = "user"
    with pytest.raises(SystemExit, match="admin"):
        control._provision_chat_ui_effort_selector("http://127.0.0.1:3000")


def test_tiered_control_requires_exact_eight_gpu_inventory() -> None:
    control = _load(EXAMPLE / "control.py", "tiered_example_control")
    text = "\n".join(
        f"{index}, NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 12.0, "
        f"00000000:{16 + index:02x}:00.0"
        for index in range(8)
    )
    rows = control._gpu_inventory(text)
    assert sorted(rows) == list(range(8))


def test_tiered_control_uses_explicit_public_ui_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    control = _load(EXAMPLE / "control.py", "tiered_example_public_host")
    monkeypatch.setenv("PUBLIC_HOST", "gpu.example.test")
    storage = {
        "qwen_models": tmp_path / "qwen-models",
        "deepseek_models": tmp_path / "deepseek-models",
        "webui": tmp_path / "webui",
        "deepseek_cache": tmp_path / "deepseek-cache",
        **{
            f"qwen_cache_{index}": tmp_path / f"qwen-cache-{index}"
            for index in range(4)
        },
    }
    monkeypatch.setattr(control, "_storage_paths", lambda: storage)

    assert control._public_ui_host() == "gpu.example.test"
    assert control._compose_env()["API_BIND_ADDRESS"] == "0.0.0.0"
    monkeypatch.setenv("API_BIND_ADDRESS", "127.0.0.1")
    assert control._compose_env()["API_BIND_ADDRESS"] == "127.0.0.1"
    assert control._compose_env()["CHAT_UI_BIND_ADDRESS"] == "0.0.0.0"
    assert control._compose_env()["DEEPSEEK_L1_PORT"] == "8005"


def test_tiered_control_rejects_persistent_storage_outside_nvme(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _load(EXAMPLE / "control.py", "tiered_example_nvme")
    monkeypatch.setenv("NVME_STORAGE_ROOT", "/tmp/not-nvme")
    with pytest.raises(SystemExit, match="/mnt/nvme"):
        control._nvme_root()


def test_tiered_verification_rejects_storage_outside_nvme(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVME_STORAGE_ROOT", "/tmp/not-nvme")
    with pytest.raises(SystemExit, match="/mnt/nvme"):
        _load(EXAMPLE / "verification.py", "tiered_example_verification_nvme")





def _write_product_serving_result(row_dir: Path, stages: list[dict]) -> None:
    result = {
        "summary": {
            "requests": 1,
            "completion_tokens_total": 32,
            "output_tokens_per_s": 1.0,
        },
        "samples": [
            {
                "completion_tokens": 32,
                "trace": {"status": "valid", "stages": stages},
            }
        ],
    }
    (row_dir / "result-serving.json").write_text(json.dumps(result))


@pytest.mark.parametrize("node", ["review", "audit"])
@pytest.mark.parametrize("failure_mode", ["missing", "failed"])
def test_tiered_product_serving_requires_every_internal_stage(
    node: str,
    failure_mode: str,
    tmp_path: Path,
) -> None:
    benchmark = _load(
        EXAMPLE / "verification.py",
        f"tiered_product_stage_{node}_{failure_mode}",
    )
    stages = [
        {"node": "head", "kind": "generation", "status": "success"},
        {
            "node": "synthesis",
            "role": "publisher",
            "kind": "generation",
            "status": "success",
        },
        *[
            {"node": name, "kind": "generation", "status": "success"}
            for name in benchmark._PRIMARY_INTERNAL_NODES
        ],
        *[
            {"node": name, "role": "verifier", "kind": "verification", "status": "success",
             "verification_pass": True, "verification_inconclusive": False}
            for name in benchmark._PRIMARY_VERIFICATION_NODES
        ],
    ]

    def validate() -> int:
        return benchmark._validate_serving_row(
            tmp_path,
            1,
            32,
            expected_route="synthesis",
            expected_role="publisher",
            require_head=True,
            expected_generation_nodes=benchmark._PRIMARY_INTERNAL_NODES,
            expected_verification_nodes=benchmark._PRIMARY_VERIFICATION_NODES,
        )

    _write_product_serving_result(tmp_path, stages)
    assert validate() == 0

    if failure_mode == "missing":
        stages = [stage for stage in stages if stage["node"] != node]
    else:
        next(stage for stage in stages if stage["node"] == node)["status"] = "failed"
    _write_product_serving_result(tmp_path, stages)

    assert validate() == 1


def test_tiered_product_serving_judged_routes_accept_one_direct_final(
    tmp_path: Path,
) -> None:
    """DTO-D13: with judged routes every sample must trace the judge and
    exactly one profile's final unit; direct routes carry no head/internal
    stage contract, while primary keeps the full dual-track contract."""

    benchmark = _load(EXAMPLE / "verification.py", "tiered_product_judged_routes")
    judge = {"node": "profile_judge", "kind": "classification", "status": "success"}
    direct = {
        "node": "qwen_think_answer",
        "role": "publisher",
        "kind": "generation",
        "status": "success",
    }
    primary = [
        {"node": "head", "kind": "generation", "status": "success"},
        {"node": "synthesis", "role": "publisher", "kind": "generation", "status": "success"},
        *[
            {"node": name, "kind": "generation", "status": "success"}
            for name in benchmark._PRIMARY_INTERNAL_NODES
        ],
        *[
            {"node": name, "role": "verifier", "kind": "verification", "status": "success",
             "verification_pass": True, "verification_inconclusive": False}
            for name in benchmark._PRIMARY_VERIFICATION_NODES
        ],
    ]

    def validate(*, require_primary=False) -> int:
        return benchmark._validate_serving_row(
            tmp_path,
            1,
            32,
            expected_route="synthesis",
            expected_role="publisher",
            require_head=True,
            expected_generation_nodes=benchmark._PRIMARY_INTERNAL_NODES,
            expected_verification_nodes=benchmark._PRIMARY_VERIFICATION_NODES,
            judged_routes=True, require_primary=require_primary,
        )

    _write_product_serving_result(tmp_path, [judge, direct])
    assert validate() == 0
    assert validate(require_primary=True) == 1
    routes = json.loads((tmp_path / "routes.json").read_text())
    assert routes["routes"] == {"qwen_think_medium": {"requests": 1, "ttft_p50_ms": None}}
    _write_product_serving_result(tmp_path, [judge, *primary])
    assert validate(require_primary=True) == 0
    # No judge stage, an ambiguous pair of finals, or a primary sample missing
    # its internal contract all fail the row.
    _write_product_serving_result(tmp_path, [direct])
    assert validate() == 1
    _write_product_serving_result(tmp_path, [judge, direct, primary[1]])
    assert validate() == 1
    _write_product_serving_result(tmp_path, [judge, *primary[:-1]])
    assert validate() == 1


def test_tiered_coding_gate_requires_fresh_completed_primary_and_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    benchmark = _load(EXAMPLE / "verification.py", "tiered_example_coding_gate")
    summaries = {
        "coding-c1": {"ttft_p50_ms": 1500.0},
        "deepseek-direct-c1": {"ttft_p50_ms": 800.0},
    }
    calls = []
    validations = []
    invalid_rows = set()

    def validate(row_dir, *_args, **kwargs):
        validations.append((row_dir.name, kwargs))
        return int(row_dir.name in invalid_rows)

    def bench(**kwargs):
        calls.append(kwargs)
        return 0

    routes = {"routes": {"primary": {"requests": 32, "ttft_p50_ms": 1500.0}},
              "judged_samples": 32, "judge_total_ms_p50": 120.0}
    monkeypatch.setattr(benchmark, "_bench_row", bench)
    monkeypatch.setattr(benchmark, "_validate_serving_row", validate)
    monkeypatch.setattr(benchmark, "_row_summary", lambda row: summaries.get(row.name))
    monkeypatch.setattr(benchmark, "_row_routes", lambda row: routes)
    monkeypatch.setitem(benchmark.SPEC["verification"]["coding"], "concurrency", [1])

    assert benchmark._coding_matrix(tmp_path / "pass", require_primary=True) == 0
    gate = json.loads((tmp_path / "pass" / "ttft-gate.json").read_text())["gates"]["1"]
    assert gate["passed"] is True
    assert gate["denominator_source"] == "fresh_completed_native_six_gpu"
    assert calls[-1]["max_tokens"] == calls[-2]["max_tokens"]
    assert calls[-1]["public_tokenizer"] is True
    assert validations[-2][1]["require_primary"] is True
    assert validations[-1][1]["public_tokens"] is True

    summaries["coding-c1"]["ttft_p50_ms"] = 1700.0
    assert benchmark._coding_matrix(tmp_path / "slow", require_primary=True) == 1
    summaries["coding-c1"]["ttft_p50_ms"] = 1500.0
    invalid_rows.add("deepseek-direct-c1")
    assert benchmark._coding_matrix(tmp_path / "incomplete", require_primary=True) == 1
    invalid_rows.clear()
    del summaries["deepseek-direct-c1"]
    assert benchmark._coding_matrix(tmp_path / "missing", require_primary=True) == 1
    summaries["deepseek-direct-c1"] = {"ttft_p50_ms": float("inf")}
    assert benchmark._coding_matrix(tmp_path / "nonfinite", require_primary=True) == 1
    summaries["deepseek-direct-c1"] = {"ttft_p50_ms": 800.0}
    routes["routes"] = {"deepseek_think": {"requests": 32, "ttft_p50_ms": 9000.0}}
    assert benchmark._coding_matrix(tmp_path / "direct-only", require_primary=True) == 1
    assert benchmark._coding_matrix(tmp_path / "natural", require_primary=False) == 0
    diagnostic = json.loads((tmp_path / "natural" / "ttft-gate.json").read_text())
    assert diagnostic["gates"]["1"]["satisfies_primary_gate"] is False


def test_primary_comparison_keeps_judge_and_restores_gateway_on_failure(monkeypatch, tmp_path):
    from types import SimpleNamespace

    benchmark = _load(EXAMPLE / "verification.py", "tiered_primary_lifecycle")
    production = (EXAMPLE / "auto-max.yaml").read_bytes()
    mounted = []

    def compose(_command):
        mounted.append(benchmark.os.environ.get("KAIRYU_ORCHESTRATOR_SPEC_PATH"))

    def routing(_url):
        configuration = yaml.safe_load(Path(mounted[-1]).read_text())
        return {"models": {"kairyu-auto-max": {
            "roles": configuration["roles"],
            "profiles": {
                profile["name"]: profile["roles"] for profile in configuration["profiles"]
            },
            "profile_judge": configuration["profile_judge"],
        }}}

    monkeypatch.delenv("KAIRYU_ORCHESTRATOR_SPEC_PATH", raising=False)
    monkeypatch.setattr(benchmark, "_control_module", lambda: SimpleNamespace(
        _compose=compose, _json_url=routing,
    ))
    with pytest.raises(RuntimeError, match="failed measurement"):
        with benchmark._primary_profile_override(tmp_path):
            override = yaml.safe_load(Path(mounted[-1]).read_text())
            original = yaml.safe_load(production)
            assert override["profile_judge"] == original["profile_judge"]
            assert all(profile["roles"] == original["roles"] for profile in override["profiles"])
            assert override["budget"] == original["budget"]
            raise RuntimeError("failed measurement")

    assert mounted == [str((tmp_path / "primary-comparison.yaml").resolve()), None]
    assert (EXAMPLE / "auto-max.yaml").read_bytes() == production
    assert json.loads((tmp_path / "primary-override.json").read_text())["restored"] is True


def test_primary_functional_evidence_checks_generated_json_and_each_choice(monkeypatch):
    from kairyu.orchestration.conductor import IntermediateOutput

    benchmark = _load(EXAMPLE / "verification.py", "tiered_primary_evidence")
    evidence = benchmark._PrimaryEvidence()
    checklist = json.dumps([{
        "id": "R1", "priority": "minimum", "requirement": "Return two sentences.",
        "acceptance_criterion": "Exactly two sentences.", "source": "user turn 1",
    }])

    def emit(node, text, choice_index=None, attempt=0):
        output = IntermediateOutput(
            node=node, role="verifier" if node == "audit" else "planner", attempt=attempt,
            worker="tier2", engine="openai", model="deepseek-v4.1-flash", text=text,
            choice_index=choice_index,
        )
        evidence.observe({"choices": [{"index": 0, "delta": {
            "reasoning_content": output.as_markdown() + "\n\n---\n\n",
        }}]})

    emit("requirements", checklist)
    for index in (0, 1):
        emit("audit", "PASS\nR1: met; evidence: both output sentences; correction: none.", index)
    events = [{"node": node, "kind": "generation", "status": "success",
               "worker": "tier1" if node.startswith("answer_") else "tier2",
               "detail": {"reasoning_effort": "max"}}
              for node in benchmark._PRIMARY_INTERNAL_NODES]
    events += [{"node": "audit", "kind": "verification", "status": "success",
                "worker": "tier2", "attempt": 0,
                "detail": {"choice_index": index, "pass": True, "inconclusive": False}}
               for index in (0, 1)]
    evidence.observe({"kairyu_trace_v2": {"events": events}})
    report = evidence.validate(effort="max", choices=2)
    assert report["requirement_ids"] == ["R1"]
    assert [row["choice_index"] for row in report["audits"]] == [0, 1]
    assert "Return two sentences." not in json.dumps(report)
    events[-1]["detail"].update({"pass": False, "refinement_exhausted": True})
    with pytest.raises(ValueError, match="configured repairs"):
        evidence.validate(effort="max", choices=2)
    events[-1]["attempt"] = 2
    emit("audit", "FAIL\nR1: unmet; evidence: one sentence; correction: add the second.",
         choice_index=1, attempt=2)
    exhausted = evidence.validate(effort="max", choices=2)
    assert exhausted["audits"][-1]["verdict"] == "FAIL"
    events.pop()
    with pytest.raises(ValueError, match="no independent DeepSeek audit"):
        evidence.validate(effort="max", choices=2)
    with pytest.raises(ValueError, match="effective high-floor/max"):
        evidence.validate(effort="low", choices=1)
    with pytest.raises(ValueError, match="duplicate object key"):
        benchmark._requirement_ids(checklist.replace('"id": "R1"', '"id":"R1","id":"R1"'))
    with pytest.raises(ValueError):
        benchmark._requirement_ids('```json\n' + checklist + '\n```')


def test_natural_completion_gate_rejects_length_or_missing_done(tmp_path):
    benchmark = _load(EXAMPLE / "verification.py", "tiered_completed_response")
    result = {
        "summary": {"requests": 1, "public_completion_tokens_total": 2,
                    "public_output_tokens_per_s": 1.0},
        "samples": [{"public_completion_tokens": 2, "stream_complete": True,
                     "finish_reasons": ["stop"]}],
    }

    def validate():
        (tmp_path / "result-serving.json").write_text(json.dumps(result))
        return benchmark._validate_serving_row(tmp_path, 1, 100, public_tokens=True)

    assert validate() == 0
    result["samples"][0]["finish_reasons"] = ["length"]
    assert validate() == 1
    result["samples"][0]["finish_reasons"] = ["stop"]
    result["samples"][0]["stream_complete"] = False
    assert validate() == 1
