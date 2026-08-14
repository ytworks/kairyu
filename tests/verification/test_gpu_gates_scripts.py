"""m19 D3: every gpu_gates script dry-runs and references REAL files/tests."""

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from kairyu.deploy.spec import load_deployment_spec

SCRIPTS = sorted(Path("scripts/gpu_gates").glob("[0-9g]*.sh"))
GATEWAY_GPU_CONFIG = Path("deploy/compose/gateway-gpu.yaml")
GPU_GATE_LIB = Path("scripts/gpu_gates/_lib.sh")
PRODUCTION_GATE = Path("scripts/gpu_gates/09_production.sh")
CHECK_SERVED_MODEL = Path("scripts/gpu_gates/check_served_model.py")


def _load_check_served_model():
    assert CHECK_SERVED_MODEL.is_file(), "production model preflight script is missing"
    spec = importlib.util.spec_from_file_location("check_served_model", CHECK_SERVED_MODEL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_dry_run_emits_commands(script):
    result = subprocess.run(
        ["bash", str(script), "--dry-run"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    commands = [line for line in result.stdout.splitlines() if line.startswith("+ ")]
    assert commands, f"{script.name} emitted no commands"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_referenced_paths_exist(script):
    """Deploy day must not discover missing files: every tests/, scripts/,
    bench/ and deploy/ path a script mentions exists TODAY."""
    text = script.read_text()
    for match in re.findall(r"(?:tests|scripts|bench|deploy)/[\w./-]+", text):
        path = Path(match)
        assert path.exists(), f"{script.name} references missing {match}"


def test_gpu_marker_tests_exist_and_are_deselected():
    gpu_tests = list(Path("tests/gpu").glob("test_*.py"))
    assert gpu_tests, "tests/gpu mirror is empty"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/gpu",
            "--collect-only",
            "-q",
            "--no-cov",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == pytest.ExitCode.NO_TESTS_COLLECTED, result.stderr
    assert "deselected" in result.stdout  # addopts excludes gpu by default


@pytest.mark.parametrize(
    ("script_name", "minimum_gpu_argument"),
    [
        ("01_runner.sh", "--min-gpus 1"),
        ("02_quant_bench.sh", "--min-gpus 1"),
        ("03_deferred.sh", "--min-gpus 4"),
        ("06_multigpu.sh", "--min-gpus 8"),
        ("g4_moe.sh", "--min-gpus 2"),
    ],
)
def test_formal_gpu_pytest_cannot_turn_selected_skips_into_success(
    script_name, minimum_gpu_argument
):
    script = Path("scripts/gpu_gates", script_name).read_text(encoding="utf-8")

    assert minimum_gpu_argument in script
    assert "pytest_no_skip " in script
    assert "run uv run pytest " not in script


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
def test_gate_scripts_use_frozen_uv_and_skip_rejecting_pytest(script):
    text = script.read_text(encoding="utf-8")

    assert "run uv run pytest " not in text
    for line in text.splitlines():
        if "uv run " in line:
            assert "uv run --frozen " in line, (script, line)


def test_deferred_gate_collects_unit_cuda_placement_tests():
    script = Path("scripts/gpu_gates/03_deferred.sh").read_text(encoding="utf-8")

    assert "--min-gpus 4" in script
    assert "--min-compute-capability 10.0" in script
    for prerequisite in (
        "--require-module flashinfer",
        "--require-module transformers",
        "--require-module triton",
        "--require-executable ninja",
        "--require-executable nvidia-smi",
    ):
        assert prerequisite in script
    assert (
        "pytest_no_skip -m gpu tests/gpu tests/unit/test_pd_factory.py -v"
        in script
    )


def test_quant_and_qwen_gates_preflight_their_actual_runtime_dependencies():
    quant = Path("scripts/gpu_gates/02_quant_bench.sh").read_text(encoding="utf-8")
    qwen = Path("scripts/gpu_gates/06_multigpu.sh").read_text(encoding="utf-8")

    assert "--min-compute-capability 10.0" in quant
    assert "--require-module flashinfer" in quant
    assert "--require-module triton" in quant
    assert "--require-module flashinfer" in qwen
    assert "--require-module transformers" in qwen
    assert "--require-executable ninja" in qwen


def test_g4_gate_preflights_transformers_before_cpu_parity():
    script = Path("scripts/gpu_gates/g4_moe.sh").read_text(encoding="utf-8")

    prerequisite = "preflight --require-module transformers"
    cpu_parity = "pytest_no_skip tests/unit/test_moe_mla_parity.py"
    assert prerequisite in script
    assert script.index(prerequisite) < script.index(cpu_parity)


def test_shared_bench_model_default_matches_gpu_gateway_pool():
    spec = load_deployment_spec(GATEWAY_GPU_CONFIG)
    match = re.search(
        r"^KAIRYU_BENCH_MODEL=\$\{KAIRYU_BENCH_MODEL:-(?P<model>[^}]+)\}$",
        GPU_GATE_LIB.read_text(),
        re.MULTILINE,
    )

    assert match is not None
    model = match.group("model")
    assert set(spec.pools) == {model}
    assert {replica.options["model"] for replica in spec.pools[model].replicas} == {
        model
    }


def test_production_gate_model_steps_use_shared_variable():
    model_args = re.findall(r"--model\s+(\S+)", PRODUCTION_GATE.read_text())

    assert model_args == ['"$KAIRYU_BENCH_MODEL"', '"$KAIRYU_BENCH_MODEL"']


def test_model_preflight_accepts_an_exact_served_id():
    checker = _load_check_served_model()

    def handler(request):
        assert request.url == httpx.URL("https://gateway.test/v1/models")
        return httpx.Response(
            200,
            json={"data": [{"id": "default-extra"}, {"id": "default"}]},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        served_ids = checker.check_served_model(
            "https://gateway.test/v1/", "default", client=client
        )

    assert served_ids == ("default-extra", "default")


def test_model_preflight_exits_nonzero_for_absent_exact_id_with_served_ids(
    monkeypatch, capsys
):
    checker = _load_check_served_model()
    client_type = httpx.Client

    def handler(_request):
        return httpx.Response(200, json={"data": [{"id": "default-extra"}]})

    monkeypatch.setattr(
        checker.httpx,
        "Client",
        lambda **_kwargs: client_type(transport=httpx.MockTransport(handler)),
    )

    exit_code = checker.main(
        ["--base-url", "https://gateway.test/v1", "--model", "default"]
    )

    message = capsys.readouterr().err
    assert exit_code == 1
    assert "requested model 'default'" in message
    assert "served model IDs: ['default-extra']" in message


@pytest.mark.parametrize(
    ("kind", "payload"),
    [
        ("invalid-json", "not-json"),
        ("non-object", []),
        ("missing-data", {}),
        ("non-list-data", {"data": "default"}),
        ("invalid-entry", {"data": [{"id": 3}]}),
    ],
)
def test_model_preflight_rejects_malformed_responses(kind, payload):
    checker = _load_check_served_model()

    def handler(_request):
        if kind == "invalid-json":
            return httpx.Response(200, text=payload)
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(checker.ModelPreflightError, match="model list response"):
            checker.check_served_model(
                "https://gateway.test/v1", "default", client=client
            )


def test_model_preflight_rejects_non_2xx_without_leaking_credentials_or_body():
    checker = _load_check_served_model()

    def handler(_request):
        return httpx.Response(503, text="api_key=server-secret")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(checker.ModelPreflightError) as exc_info:
            checker.check_served_model(
                "https://user:client-secret@gateway.test/v1",
                "default",
                client=client,
            )

    message = str(exc_info.value)
    assert "HTTP 503" in message
    assert "client-secret" not in message
    assert "server-secret" not in message


def test_production_gate_preflights_shared_model_after_ready_before_benchmark():
    commands = [
        line.strip()
        for line in PRODUCTION_GATE.read_text().splitlines()
        if line.startswith("run ")
    ]
    ready = "run curl -sf http://127.0.0.1:8000/readyz"
    preflight = (
        "run uv run --frozen python scripts/gpu_gates/check_served_model.py "
        '--base-url http://127.0.0.1:8000/v1 --model "$KAIRYU_BENCH_MODEL"'
    )
    benchmark = (
        "run uv run --frozen python verification/l1/performance/serving_bench.py "
        '--base-url http://127.0.0.1:8000/v1 --model "$KAIRYU_BENCH_MODEL"'
    )

    assert ready in commands
    assert preflight in commands
    assert benchmark in commands
    assert commands.index(ready) < commands.index(preflight) < commands.index(benchmark)


@pytest.mark.parametrize("model", ["default", "override-model"])
def test_production_gate_dry_run_preflights_same_model_before_benchmark(model):
    env = os.environ.copy()
    env["KAIRYU_BENCH_MODEL"] = model
    result = subprocess.run(
        ["bash", str(PRODUCTION_GATE), "--dry-run"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    commands = [line for line in result.stdout.splitlines() if line.startswith("+ ")]
    ready = "+ curl -sf http://127.0.0.1:8000/readyz"
    preflight = (
        "+ uv run --frozen python scripts/gpu_gates/check_served_model.py "
        f"--base-url http://127.0.0.1:8000/v1 --model {model}"
    )
    benchmark = (
        "+ uv run --frozen python verification/l1/performance/serving_bench.py "
        f"--base-url http://127.0.0.1:8000/v1 --model {model}"
    )

    assert commands.index(ready) < commands.index(preflight) < commands.index(benchmark)
