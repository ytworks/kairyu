"""Regression coverage for GPU gate integrity issues #103 and #104."""

import importlib.util
import json
from pathlib import Path

RECORD_ENV = Path("scripts/gpu_gates/record_env.py")


def _load_record_env():
    spec = importlib.util.spec_from_file_location("record_env", RECORD_ENV)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_multigpu_gate_invokes_only_real_nccl_coverage():
    script = Path("scripts/gpu_gates/06_multigpu.sh").read_text()

    assert "KAIRYU_DIST_BACKEND" not in script
    assert script.count("tests/dist -v") == 1
    assert script.count("tests/gpu/test_moe_parallel_nccl.py -v") == 1


def test_environment_gate_writes_schema_valid_record(tmp_path, monkeypatch):
    recorder = _load_record_env()
    profile = recorder.HardwareProfile(
        arch="cuda",
        device_name="Fake GPU",
        sm=120,
        device_count=2,
        memory_gb=96.0,
        formats=("bf16", "fp8", "nvfp4"),
        interconnect="pcie",
    )
    outputs = {
        ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"): (
            "595.71.05\n595.71.05"
        ),
        ("nvidia-smi", "topo", "-m"): "GPU0 GPU1\nGPU0 X NODE\nGPU1 NODE X",
        (
            "nvidia-smi",
            "--query-gpu=index,name,pci.bus_id,memory.total,mig.mode.current,vbios_version",
            "--format=csv,noheader",
        ): "0, Fake GPU, 0000:01:00.0, 98304 MiB, Disabled, fake-vbios",
    }
    monkeypatch.setattr(recorder, "_version", lambda name: f"{name}-version")

    record = recorder.build_env_record(
        profile=profile,
        record_date="2026-07-23",
        run_command=lambda command: outputs[tuple(command)],
    )
    path = recorder.write_env_record(record, tmp_path)
    payload = json.loads(path.read_text())

    assert path.name == "env-2026-07-23.json"
    assert payload["driver"] == "595.71.05"
    assert payload["profile"]["sm"] == 120
    assert payload["profile"]["p2p_matrix"] is None
    assert payload["library_versions"]["flashinfer"] == "flashinfer-python-version"
    assert "Numeric bandwidth/P2P measurements are unmeasured" in payload["notes"]
    assert "fake-vbios" in payload["notes"]
