from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
ENVIRONMENTS = (
    "qwen3.6-27b-1gpu",
    "deepseek-v4-flash-0731-8gpu",
    "qwen3.6-deepseek-v4-8gpu",
)
QWEN_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
DEEPSEEK_REVISION = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"


def test_examples_tree_contains_only_frontier_environments_and_shared_code() -> None:
    directories = {
        path.name for path in EXAMPLES.iterdir() if path.is_dir()
    }
    assert directories == {"_shared", *ENVIRONMENTS}
    for name in ENVIRONMENTS:
        environment = EXAMPLES / name
        assert {
            "README.md",
            "bench.sh",
            "compose.yaml",
            "example.json",
            "run.sh",
        } <= {path.name for path in environment.iterdir()}


def test_example_manifests_pin_images_revisions_and_native_contexts() -> None:
    observed: dict[str, int] = {}
    for name in ENVIRONMENTS:
        raw = json.loads((EXAMPLES / name / "example.json").read_text())
        assert raw["environment"] == name
        assert all("@sha256:" in image for image in raw["images"].values())
        assert raw["minimum_compute_capability"] == 12.0
        for model in raw["models"]:
            observed[model["revision"]] = model["max_context_tokens"]
    assert observed == {
        QWEN_REVISION: 262_144,
        DEEPSEEK_REVISION: 1_048_576,
    }


def test_new_configs_never_enable_legacy_chat_or_shrink_context() -> None:
    serialized = "\n".join(
        path.read_text(encoding="utf-8")
        for name in ENVIRONMENTS
        for path in (EXAMPLES / name).glob("*.yaml")
    )
    assert "legacy_chat_models" not in serialized
    for name in ENVIRONMENTS:
        for path in (EXAMPLES / name).glob("*.yaml"):
            raw = yaml.safe_load(path.read_text())
            for engine in (raw.get("engines") or {}).values():
                configured = (engine.get("options") or {}).get("max_model_len")
                assert configured not in {32_768, "32768"}
    assert "execution_mode: native" in serialized
    assert "max_model_len: 262144" in serialized
    assert "max_model_len: 1048576" in serialized
    assert "mtp_enabled: false" in serialized
    assert "dspark_enabled: false" in serialized


def test_orchestration_router_digest_matches_every_deployment() -> None:
    environment = EXAMPLES / "qwen3.6-deepseek-v4-8gpu"
    artifact = environment / "router.json"
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert digest == "9b3cc02780fd5a1cdf1073557b66900e14dd59b6dd655b961f6f85bc3cb51ca5"
    for path in environment.glob("auto-*.yaml"):
        raw = yaml.safe_load(path.read_text())
        assert raw["router"]["sha256"] == digest


def test_example_shell_entrypoints_are_syntactically_valid() -> None:
    scripts = [
        *(EXAMPLES / "_shared").glob("*.sh"),
        *(path for name in ENVIRONMENTS for path in (EXAMPLES / name).glob("*.sh")),
    ]
    assert scripts
    for path in scripts:
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_benchmark_cli_lists_required_ids() -> None:
    for name in ENVIRONMENTS:
        result = subprocess.run(
            [str(EXAMPLES / name / "bench.sh"), "list"],
            check=True,
            text=True,
            capture_output=True,
            cwd=ROOT,
        )
        ids = set(result.stdout.splitlines())
        assert {"accuracy", "structured", "long-context", "serving"} <= ids
        assert any(item.startswith("accuracy:") for item in ids)
        assert any(item.startswith("long-context:") for item in ids)
        assert ("orchestration" in ids) == (name == ENVIRONMENTS[-1])


def test_each_benchmark_attempt_finalizes_reports(monkeypatch, tmp_path) -> None:
    module_spec = importlib.util.spec_from_file_location(
        "frontier_benchctl",
        EXAMPLES / "_shared/benchctl.py",
    )
    assert module_spec is not None and module_spec.loader is not None
    benchctl = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(benchctl)
    environment = EXAMPLES / ENVIRONMENTS[0]
    manifest = json.loads((environment / "example.json").read_text())
    monkeypatch.setattr(
        benchctl,
        "_evidence",
        lambda _spec_dir, _spec, _backend, output, _port: output.mkdir(
            parents=True, exist_ok=True
        ),
    )
    monkeypatch.setattr(
        benchctl,
        "_quality_command",
        lambda *_args: ["benchmark-fixture"],
    )

    for returncode, expected in ((0, "passed"), (7, "failed")):
        monkeypatch.setattr(
            benchctl.subprocess,
            "run",
            lambda *_args, _returncode=returncode, **_kwargs: SimpleNamespace(
                returncode=_returncode
            ),
        )
        run_root = tmp_path / str(returncode)
        passed = benchctl._run_one(
            environment,
            manifest,
            "kairyu",
            "structured",
            run_root,
        )
        report_dir = run_root / "structured"
        payload = json.loads((report_dir / "report.json").read_text())
        assert passed is (returncode == 0)
        assert payload["status"] == expected
        assert (report_dir / "report.md").is_file()
        assert (report_dir / "raw.log").is_file()


def test_benchmark_startup_failure_still_finalizes_report(monkeypatch, tmp_path) -> None:
    module_spec = importlib.util.spec_from_file_location(
        "frontier_benchctl_startup",
        EXAMPLES / "_shared/benchctl.py",
    )
    assert module_spec is not None and module_spec.loader is not None
    benchctl = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(benchctl)
    manifest = json.loads(
        (EXAMPLES / ENVIRONMENTS[0] / "example.json").read_text()
    )
    monkeypatch.setattr(benchctl, "_root", lambda _path: tmp_path)
    monkeypatch.setattr(
        benchctl,
        "_start",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("readiness failed")),
    )

    run_root, passed = benchctl._backend_run(
        EXAMPLES / ENVIRONMENTS[0], manifest, "kairyu", "accuracy"
    )

    assert passed is False
    assert json.loads((run_root / "integrated.json").read_text())["passed"] is False
    assert (run_root / "startup/report.json").is_file()
    assert "readiness failed" in (run_root / "startup/report.md").read_text()
