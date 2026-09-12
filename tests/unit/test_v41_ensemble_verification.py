from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/qwen3.8-deepseek-v4.1-8gpu"


def load(name):
    path = EXAMPLE / f"{name}.py"
    assert path.exists(), "new verification command is missing"
    spec = importlib.util.spec_from_file_location(f"v41_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("missing", ["requirements", "answer_1", "answer_2", "critique", "audit"])
def test_serving_gate_rejects_missing_required_stage(tmp_path, missing):
    module = load("verification")
    nodes = ("draft", "requirements", "policies", "answer_1", "answer_2", "critique")
    stages = [
        {"node": "head", "kind": "generation", "status": "success"},
        {"node": "synthesis", "role": "publisher", "kind": "generation", "status": "success"},
        *[{"node": name, "kind": "generation", "status": "success"} for name in nodes],
        {"node": "audit", "role": "verifier", "kind": "verification", "status": "success"},
    ]

    def verify(stages):
        (tmp_path / "result-serving.json").write_text(
            json.dumps(
                {
                    "summary": {
                        "requests": 1,
                        "completion_tokens_total": 32,
                        "output_tokens_per_s": 1.0,
                    },
                    "samples": [
                        {"completion_tokens": 32, "trace": {"status": "valid", "stages": stages}}
                    ],
                }
            )
        )
        return module._validate_serving_row(
            tmp_path,
            1,
            32,
            expected_route="synthesis",
            expected_role="publisher",
            require_head=True,
            expected_generation_nodes=module._DUAL_TRACK_INTERNAL_NODES,
            expected_verification_nodes=module._DUAL_TRACK_VERIFICATION_NODES,
        )

    assert verify(stages) == 0
    assert verify([stage for stage in stages if stage["node"] != missing]) == 1


def test_image_diagnostic_rejects_legacy_description_stage():
    module = load("requirements_quality")
    cases = json.loads((EXAMPLE / "requirements-quality-cases.json").read_text())["cases"]
    case = next(case for case in cases if case["id"] == "image-requirements")
    result = {"trace": {"events": []}, "reasoning": "", "answer": ""}
    report = module.validate_result(case, result, requirement_cap=8192)
    assert report["contract"]["checks"]["native_image_path_only"] is True
    assert report["contract"]["passed"] is False
    result["trace"]["events"] = [{"node": "image_description"}]
    report = module.validate_result(case, result, requirement_cap=8192)
    assert report["contract"]["checks"]["native_image_path_only"] is False
    assert report["semantic_review"] == "not_performed_by_this_evaluator"


def test_served_config_hash_includes_shared_runtime(monkeypatch, tmp_path):
    module = load("verification")
    for name in module.CONFIG_FILES:
        target = tmp_path / "example" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((EXAMPLE / name).read_bytes())
    monkeypatch.setattr(module, "HERE", tmp_path / "example")
    original = module._served_config_sha256()
    shared = tmp_path / "deepseek-v4.1-flash-8gpu/patch_runtime.py"
    shared.write_bytes(shared.read_bytes() + b"\n# provenance test\n")
    assert module._served_config_sha256() != original


def test_paired_baseline_uses_native_nonthinking_role(tmp_path):
    module = load("verification")
    hooks = load("requirements_budget")
    source = tmp_path / "coding.json"
    target = tmp_path / "direct.json"
    source.write_text(json.dumps([{"conversations": [{"from": "human", "value": "Solve x"}]}]))
    module._direct_dataset(source, target)
    prompt = json.loads(target.read_text())[0]["conversations"][0]["value"]
    middleware = hooks.RequirementsBudgetMiddleware(None, config_dir=EXAMPLE)
    changed, role = middleware.transform(
        {
            "model": "deepseek-v4.1-flash",
            "messages": [{"role": "user", "content": prompt}],
        }
    )
    assert role == "deepseek_answer"
    assert changed["chat_template_kwargs"]["thinking"] is False
    assert changed["chat_template_kwargs"]["enable_thinking"] is False
    assert "Solve x" in prompt


def test_ttft_gate_has_no_borrowed_baseline(monkeypatch, tmp_path):
    module = load("verification")
    monkeypatch.setattr(module, "_bench_row", lambda **kwargs: 0)
    monkeypatch.setattr(module, "_validate_serving_row", lambda *a, **k: 0)
    monkeypatch.setattr(module, "_row_routes", lambda _: None)
    monkeypatch.setattr(
        module, "_row_summary", lambda p: {"ttft_p50_ms": 100.0} if p.name == "coding-c1" else None
    )
    module.SPEC["verification"]["coding"]["concurrency"] = [1]
    assert module.serving_auto_max_coding(tmp_path) == 1


@pytest.mark.parametrize(
    ("direct_code", "direct_ttft", "expected"),
    [
        (7, 100.0, 7),
        (0, None, 1),
        (0, float("inf"), 1),
        (0, float("nan"), 1),
        (0, 0, 1),
        (0, -1, 1),
        (0, True, 1),
        (0, 100.0, 0),
    ],
)
def test_ungated_coding_row_still_requires_usable_paired_baseline(
    monkeypatch, tmp_path, direct_code, direct_ttft, expected
):
    module = load("verification")
    module.SPEC["verification"]["coding"]["concurrency"] = [1]
    calls = []

    def bench(**kwargs):
        target = kwargs["results_dir"]
        calls.append(target.name)
        target.mkdir()
        value = direct_ttft if target.name.startswith("deepseek-direct") else 150.0
        (target / "result-serving.json").write_text(json.dumps({"summary": {"ttft_p50_ms": value}}))
        return direct_code if target.name.startswith("deepseek-direct") else 0

    monkeypatch.setattr(module, "_bench_row", bench)
    monkeypatch.setattr(module, "_validate_serving_row", lambda *a, **k: 0)
    monkeypatch.setattr(
        module,
        "_row_routes",
        lambda _: {"routes": {"deepseek_think": {"requests": 32, "ttft_p50_ms": 150.0}}},
    )
    assert module.serving_auto_max_coding(tmp_path) == expected
    assert calls == ["warmup", "coding-c1", "deepseek-direct-c1"]
    gate = tmp_path / "ttft-gate.json"
    if expected:
        assert not gate.exists(), "A missing baseline must not become a successful N/A row"
    else:
        row = json.loads(gate.read_text())["gates"]["1"]
        assert row["status"] == "not_applicable"
        assert row["deepseek_direct_ttft_p50_ms"] == 100.0
