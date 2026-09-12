from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/qwen3.8-deepseek-v4.1-8gpu"


def load_control():
    path = EXAMPLE / "control.py"
    assert path.is_file(), "new ensemble lifecycle is missing"
    spec = importlib.util.spec_from_file_location("v41_ensemble_control", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_status_paths_do_not_create_storage(monkeypatch, tmp_path):
    control = load_control()
    monkeypatch.setattr(control, "_nvme_root", lambda: tmp_path)
    paths = control._storage_paths()
    assert len([key for key in paths if key.startswith("qwen_cache_")]) == 2
    assert not any(tmp_path.iterdir())
    assert paths["deepseek_models"].name == "models"
    assert "deepseek-v4.1-flash-8gpu" in str(paths["deepseek_models"])


def test_compose_status_does_not_generate_secrets(monkeypatch, tmp_path):
    control = load_control()
    monkeypatch.setattr(control, "_nvme_root", lambda: tmp_path)
    monkeypatch.delenv("KAIRYU_RESPONSES_COMPACTION_SECRET", raising=False)
    monkeypatch.setenv("COMPOSE_FILE", "/unexpected/compose.yaml")
    env = control._compose_env()
    assert "COMPOSE_FILE" not in env
    assert env["COMPOSE_PROJECT_NAME"] == "qwen3-8-deepseek-v4-1-8gpu"
    assert env["API_PORT"] == "8008"
    assert env["CHAT_UI_PORT"] == "3008"
    assert env["DEEPSEEK_L1_PORT"] == "8009"
    assert not any(tmp_path.iterdir())


def test_preflight_uses_declared_gpu_allocation(monkeypatch):
    control = load_control()
    rows = "\n".join(
        f"{i}, NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 12.0, 0000:{i:02x}:00.0"
        for i in range(8)
    )
    monkeypatch.setattr(control.shutil, "which", lambda _: "/bin/executable")
    monkeypatch.setattr(control, "_run", lambda *a, **kw: subprocess.CompletedProcess(a, 0, rows))
    monkeypatch.setattr(control, "_numa_cpuset", lambda bus: str(int(bus.split(":")[1], 16)))
    env = {}
    control._preflight(env)
    assert env["QWEN_0_CPUSET"] == "6"
    assert env["QWEN_1_CPUSET"] == "7"
    assert env["DEEPSEEK_CPUSET"] == "0,1,2,3,4,5"
    assert "QWEN_2_CPUSET" not in env


def test_wrong_runtime_image_fails_before_serving(monkeypatch):
    control = load_control()
    monkeypatch.setattr(control, "_image_id", lambda _: "sha256:wrong")
    source = control.SPEC["vllm"]["deepseek"]
    with pytest.raises(SystemExit, match="image.*ID"):
        control._ensure_vllm_image(
            {"DEEPSEEK_VLLM_IMAGE": source["image"]}, "DEEPSEEK_VLLM_IMAGE", source
        )


def test_config_digest_tracks_native_hook(monkeypatch, tmp_path):
    control = load_control()
    for name in control.CONFIG_FILES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((EXAMPLE / name).read_bytes())
    monkeypatch.setattr(control, "HERE", tmp_path)
    initial = control._requirements_config_sha256()
    target = tmp_path / "requirements_budget.py"
    target.write_bytes(target.read_bytes() + b"\n# digest probe\n")
    assert control._requirements_config_sha256() != initial


@pytest.mark.parametrize("stale", [False, True])
def test_runtime_attestation_rejects_stale_containers(monkeypatch, stale):
    control = load_control()
    digest = control._requirements_config_sha256()
    project = control.SPEC["environment"].replace(".", "-")
    containers = []
    for service in ("deepseek", "qwen-0", "qwen-1", "kairyu"):
        image = (
            "sha256:api-image"
            if service == "kairyu"
            else control.SPEC["vllm"]["deepseek" if service == "deepseek" else "qwen"]["image_id"]
        )
        containers.append(
            {
                "Name": f"/{project}-{service}-1",
                "Image": image,
                "State": {"Running": True},
                "Config": {
                    "Env": [
                        "KAIRYU_REQUIREMENTS_CONFIG_SHA256=" + ("old" if stale else digest),
                        "KAIRYU_RESPONSES_COMPACTION_SECRET=must-not-appear-in-report",
                    ]
                },
            }
        )
    monkeypatch.setattr(
        control, "_run", lambda *a, **k: subprocess.CompletedProcess(a, 0, json.dumps(containers))
    )
    if stale:
        with pytest.raises(SystemExit, match="configuration.*mismatch"):
            control.runtime_attestation()
    else:
        report = control.runtime_attestation()
        assert report["config_sha256"] == digest
        assert report["containers"]["kairyu"]["image_id"] == "sha256:api-image"
        assert "must-not-appear" not in json.dumps(report)
