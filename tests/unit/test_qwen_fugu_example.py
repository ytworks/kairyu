"""Static contract for the Qwen3-32B Fugu-suite example scripts.

The scripts cannot be executed here (they need GPUs and a served model), so the
properties that would silently produce a wrong or misleading number are pinned
statically: the model is preflighted by exact id, a subset run announces itself,
and the accuracy report is what the operator is pointed at.
"""

import re
import stat
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "qwen3-32b-multi-gpu"


def _run_args(**overrides):
    """Minimal `kairyu bench run` namespace for config assembly."""
    import argparse

    defaults = dict(
        config=None,
        base_url="http://gw/v1",
        model=["qwen3-32b"],
        target=None,
        api_key_env=None,
        no_vision=False,
        reasoning_effort=None,
        top_p=None,
        sampling_seed=None,
        extra_body=None,
        suite=None,
        only=None,
        exclude=None,
        limit=None,
        attempts=None,
        smoke=False,
        offline_fixtures=False,
        seed=None,
        judge_base_url=None,
        judge_model=None,
        judge_api_key_env=None,
        judge_reasoning_effort=None,
        judge_extra_body=None,
        concurrency=None,
        results_dir=None,
        run_id=None,
        rerun=False,
        cache_dir=None,
        no_download=False,
        no_progress=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)
FUGU = EXAMPLE / "fugu-benchmark.sh"
RUN_FUGU = EXAMPLE / "run-fugu-benchmark.sh"


@pytest.fixture(scope="module")
def fugu_text() -> str:
    return FUGU.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def run_fugu_text() -> str:
    return RUN_FUGU.read_text(encoding="utf-8")


@pytest.mark.parametrize("script", [FUGU, RUN_FUGU])
def test_scripts_are_executable_posix_sh(script):
    assert script.is_file(), script
    assert script.read_text(encoding="utf-8").startswith("#!/bin/sh")
    assert stat.S_IMODE(script.stat().st_mode) & stat.S_IXUSR
    # fail fast on unset variables and errors, like the sibling scripts
    assert "set -eu" in script.read_text(encoding="utf-8")


def test_readiness_is_waited_for_before_benchmarking(fugu_text, run_fugu_text):
    for text in (fugu_text, run_fugu_text):
        assert "/readyz" in text


def test_served_model_is_preflighted_by_exact_id(fugu_text):
    """A healthy gateway can pass readyz while serving a different model."""
    assert "/models" in fugu_text
    # ids come out of the response, then compare as STRINGS: MODEL must never be
    # interpolated into the pattern, or the "exact id" check is a glob
    assert '[ "$served_id" = "$model" ]' in fugu_text
    assert "${model}" not in fugu_text.split("served_ids=")[1].split("found=0")[0]
    # and the run stops rather than benchmarking the wrong deployment
    assert re.search(r"is not served at.*\n.*exit 1", fugu_text, re.S)


def test_request_contract_is_preflighted_before_benchmarking(fugu_text):
    preflight = fugu_text.index("Qwen chat-template/generation preflight")
    benchmark = fugu_text.index("kairyu bench run")

    assert preflight < benchmark
    assert '"chat_template_kwargs"' in fugu_text
    assert '\"enable_thinking\":true' in fugu_text
    assert '"${base_url}/chat/completions"' in fugu_text
    assert "--max-time 300" in fugu_text


def test_judge_defaults_to_the_same_gateway(fugu_text):
    """The tau user simulator must share one OPENAI_BASE_URL with the target."""
    assert "--judge-base-url" in fugu_text
    assert 'judge_model="${JUDGE_MODEL:-$model}"' in fugu_text


def test_subset_and_full_runs_are_both_announced(fugu_text):
    """A capped run must never be mistaken for a full-suite number."""
    assert "SUBSET RUN" in fugu_text
    assert "FULL RUN" in fugu_text
    assert 'bench_limit="${BENCH_LIMIT:-20}"' in fugu_text
    assert "--limit" in fugu_text


def test_fixture_mode_says_its_scores_are_meaningless(fugu_text):
    assert "OFFLINE FIXTURES" in fugu_text
    assert "not meaningful" in fugu_text
    assert "--offline-fixtures" in fugu_text


def test_fugu_conditions_are_reachable_from_the_environment(fugu_text):
    for flag in (
        "--extra-body",
        "--judge-extra-body",
        "--attempts",
        "--only",
        "--exclude",
    ):
        assert flag in fugu_text, flag


def test_operator_is_pointed_at_both_artifacts(fugu_text):
    assert "scoreboard.md" in fugu_text
    assert "comparison.md" in fugu_text


def test_dataset_extra_is_installed_for_the_run(fugu_text):
    """The serving image has no dataset deps; the suite runs on the host."""
    assert "--extra bench" in fugu_text
    assert "--extra bench-agentic" in fugu_text
    assert "--with \"$tau_requirement\"" in fugu_text
    assert "fc0055dc4e0a316c3f83133267fbd6faaa770992" in fugu_text
    assert "TAU2_DATA_DIR" in fugu_text
    assert "banking_knowledge" in fugu_text
    assert "command -v uv" in fugu_text


def test_only_required_operator_setting_is_hf_token(fugu_text):
    assert 'if [ -z "${HF_TOKEN:-}" ]' in fugu_text
    assert "export HF_TOKEN" in fugu_text
    assert 'HF_TOKEN="$HF_TOKEN" PORT="$port" ./run.sh --detach' in fugu_text
    assert ".env file" in fugu_text
    assert "./run.sh --detach" in fugu_text


def test_generated_code_uses_an_immutable_docker_runner(fugu_text):
    assert "deploy/bench/Dockerfile.exec" in fugu_text
    assert "docker image inspect --format '{{.Id}}'" in fugu_text
    assert "--exec-runner docker" in fugu_text
    assert '--exec-image "$exec_image"' in fugu_text


def test_qwen_sampling_defaults_use_supported_template_controls(fugu_text):
    assert "--reasoning-effort" not in fugu_text
    assert "--judge-reasoning-effort" not in fugu_text
    assert "extra_body_default=" in fugu_text
    assert "judge_extra_body_default=" in fugu_text
    assert '"enable_thinking":true' in fugu_text
    assert '"enable_thinking":false' in fugu_text


def test_one_command_entry_point_starts_then_benchmarks(run_fugu_text):
    assert "./run.sh --detach" in run_fugu_text
    assert "exec ./fugu-benchmark.sh" in run_fugu_text
    # an already-running service is reused rather than restarted
    assert "already ready" in run_fugu_text


def test_text_only_target_is_declared_text_only(fugu_text):
    """Qwen3-32B is a causal LM; the vision family is the separate Qwen3-VL.

    Declaring it vision-capable would let CharXiv and HLE image rows be measured
    on prompts whose image parts the text-only chat template drops.
    """
    assert "--no-vision" in fugu_text
    assert 'vision_flag="--no-vision"' in fugu_text
    assert "VISION" in fugu_text  # an opt-in for a genuinely multimodal deployment
    assert "$vision_flag" in fugu_text


def test_no_vision_flag_narrows_the_target():
    from kairyu.bench.config import build_config

    assert build_config(_run_args(no_vision=True)).targets[0].supports_vision is False
    assert build_config(_run_args()).targets[0].supports_vision is True


def test_vision_slots_skip_on_a_text_only_target(tmp_path):
    """The honest outcome: skipped, not a score from an image-free prompt."""
    import httpx

    from kairyu.bench.adapters.base import RunContext
    from kairyu.bench.adapters.charxiv import CharXivAdapter
    from kairyu.bench.adapters.hle import HleAdapter
    from kairyu.bench.cache import BenchCache
    from kairyu.bench.judge import JudgeClient
    from kairyu.bench.types import BenchItem, BenchTarget, JudgeConfig, SkipItem

    target = BenchTarget(base_url="http://gw/v1", model="qwen3-32b", supports_vision=False)
    ctx = RunContext(
        cache=BenchCache(tmp_path / "cache"),
        http_factory=lambda: httpx.AsyncClient(),
        offline_fixtures=True,
        # a judge is configured, so vision is the only unmet precondition left
        judge=JudgeClient(
            JudgeConfig(base_url="http://gw/v1", model="j"),
            http_factory=lambda: httpx.AsyncClient(),
        ),
    )
    assert "vision" in CharXivAdapter().check_preconditions(target, ctx)

    image_item = BenchItem(
        id="x",
        payload={
            "question": "Q",
            "answer": "A",
            "answer_type": "multipleChoice",
            "image": "data:image/png;base64,AAAA",
        },
    )
    assert isinstance(HleAdapter().build_request(image_item, target, ctx), SkipItem)


def test_subset_warning_is_said_to_survive_into_the_artifacts(fugu_text):
    assert "scoreboard.md and comparison.md" in fugu_text
    assert "withhold every delta" in fugu_text


def test_port_reaches_the_compose_mapping(run_fugu_text):
    """PORT=9000 must not start a healthy service on 8001 and then time out."""
    compose = (EXAMPLE / "compose.yaml").read_text(encoding="utf-8")
    assert '"127.0.0.1:${PORT:-8001}:8000"' in compose
    assert 'export PORT="$port"' in run_fugu_text
    assert "export PORT" in (EXAMPLE / "run.sh").read_text(encoding="utf-8")


def test_kv_capacity_covers_the_declared_eight_way_quality_run():
    import yaml

    config = yaml.safe_load((EXAMPLE / "kairyu.template.yaml").read_text())
    options = config["engines"]["qwen3-32b"]["options"]

    assert options["num_pages"] * 16 >= 8 * 8192
    assert options["cuda_graph_max_pages"] * 16 >= 8192


def test_compose_uses_the_hardware_selected_attention_backend_by_default():
    compose = (EXAMPLE / "compose.yaml").read_text(encoding="utf-8")

    assert "KAIRYU_ATTENTION_BACKEND: ${KAIRYU_ATTENTION_BACKEND:-}" in compose
    assert "KAIRYU_ATTENTION_BACKEND:-torch" not in compose


def test_compose_serves_with_the_checkpoint_owned_qwen_chat_template():
    compose = (EXAMPLE / "compose.yaml").read_text(encoding="utf-8")
    config = (EXAMPLE / "kairyu.template.yaml").read_text(encoding="utf-8")

    assert "/models/qwen3-32b/tokenizer_config.json" in compose
    assert '.get("chat_template")' in compose
    assert "has no string chat_template" in compose
    assert "/tmp/qwen3-chat-template.jinja" in compose
    assert "qwen3-32b: /tmp/qwen3-chat-template.jinja" in config


def test_readme_documents_the_quality_suite():
    readme = (EXAMPLE / "README.md").read_text(encoding="utf-8")
    assert "run-fugu-benchmark.sh" in readme
    assert "BENCH_LIMIT" in readme
    assert "comparison.md" in readme


def test_auto_gateway_serves_distinct_conductor_and_moa_tiers():
    import yaml

    gateway = yaml.safe_load((EXAMPLE / "auto-gateway.yaml").read_text())
    standard = yaml.safe_load((EXAMPLE / "auto-orchestrator.yaml").read_text())
    maximum = yaml.safe_load((EXAMPLE / "auto-max-orchestrator.yaml").read_text())

    assert set(gateway["orchestrators"]) == {"kairyu-auto", "kairyu-auto-max"}
    assert standard.get("moa_samples", 0) == 0
    assert maximum["moa_samples"] == 3
    assert standard["shared_prefix"] == maximum["shared_prefix"]


# -- runtime boundaries (executed, not only grepped) ---------------------------


def _stub_tools(tmp_path: Path) -> Path:
    """A PATH where uv/docker are harmless and record benchmark invocation."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    marker = tmp_path / "uv-was-invoked"
    (bindir / "uv").write_text(
        f"#!/bin/sh\ntouch {marker}\nexit 0\n", encoding="utf-8"
    )
    (bindir / "uv").chmod(0o755)
    (bindir / "docker").write_text(
        "#!/bin/sh\n"
        "if [ \"${1:-}\" = info ]; then exit 0; fi\n"
        "if [ \"${1:-}\" = image ] && [ \"${2:-}\" = inspect ]; then\n"
        "  case \" $* \" in\n"
        "    *\" {{.Id}} \"*) printf '%s\\n' "
        "sha256:0000000000000000000000000000000000000000000000000000000000000000;;\n"
        "  esac\n"
        "  exit 0\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    (bindir / "docker").chmod(0o755)
    (bindir / "git").write_text(
        "#!/bin/sh\n"
        "if [ \"${1:-}\" = clone ]; then\n"
        "  for destination do :; done\n"
        "  mkdir -p \"$destination/.git\" "
        "\"$destination/data/tau2/domains/banking_knowledge\"\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"${1:-}\" = -C ] && [ \"${3:-}\" = rev-parse ]; then\n"
        "  printf '%s\\n' fc0055dc4e0a316c3f83133267fbd6faaa770992\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    (bindir / "git").chmod(0o755)
    return bindir


def _run_script(
    tmp_path: Path,
    env: dict[str, str],
    *,
    port: int | None = None,
    provision_tau_data: bool = True,
):
    """Run fugu-benchmark.sh with a stub uv; returns (CompletedProcess, marker)."""
    import os
    import subprocess

    bindir = _stub_tools(tmp_path)
    marker = tmp_path / "uv-was-invoked"
    environ = dict(os.environ)
    environ["PATH"] = f"{bindir}{os.pathsep}{environ['PATH']}"
    environ["HF_TOKEN"] = "hf_test"
    if provision_tau_data:
        tau_data = tmp_path / "tau-data"
        (tau_data / "tau2" / "domains" / "banking_knowledge").mkdir(parents=True)
        environ["TAU2_DATA_DIR"] = str(tau_data)
    else:
        environ.pop("TAU2_DATA_DIR", None)
    environ.update(env)
    if port is not None:
        environ["PORT"] = str(port)
    completed = subprocess.run(
        ["sh", str(FUGU)],
        capture_output=True,
        text=True,
        timeout=120,
        env=environ,
        check=False,
    )
    return completed, marker


def test_missing_hf_token_fails_before_setup(tmp_path):
    completed, marker = _run_script(tmp_path, {"HF_TOKEN": ""})
    assert completed.returncode == 2
    assert "HF_TOKEN is required" in completed.stderr
    assert not marker.exists()


def test_tau_task_data_checkout_is_provisioned_automatically(tmp_path):
    server = _serve_models(["qwen3-32b"])
    try:
        cache = tmp_path / "cache"
        completed, marker = _run_script(
            tmp_path,
            {
                "KAIRYU_BENCH_CACHE": str(cache),
                "BENCH_LIMIT": "1",
                "RESULTS_DIR": str(tmp_path / "results"),
            },
            port=server.server_address[1],
            provision_tau_data=False,
        )
        assert completed.returncode == 0, completed.stderr
        checkout = (
            cache
            / "harnesses"
            / "tau2-fc0055dc4e0a316c3f83133267fbd6faaa770992"
        )
        assert (checkout / "data/tau2/domains/banking_knowledge").is_dir()
        assert marker.exists()
    finally:
        server.shutdown()


@pytest.mark.parametrize("value", ["invalid", "-5", "1.5", "20x", " "])
def test_a_bad_bench_limit_never_reaches_the_benchmark(tmp_path, value):
    """`[ "$x" -gt 0 ]` returns 2 on a non-integer, and inside an `if` that does
    not trip `set -e`: an unvalidated typo would fall through to the else branch
    and launch a FULL RUN of tens of thousands of judged items."""
    completed, marker = _run_script(tmp_path, {"BENCH_LIMIT": value})
    assert completed.returncode != 0
    assert "BENCH_LIMIT must be a non-negative integer" in completed.stderr
    assert "FULL RUN" not in completed.stdout
    assert not marker.exists(), "the benchmark must not have been launched"


@pytest.mark.parametrize("name", ["ATTEMPTS", "BENCH_CONCURRENCY", "PORT"])
def test_other_numeric_inputs_are_validated_too(tmp_path, name):
    completed, marker = _run_script(tmp_path, {name: "nope"})
    assert completed.returncode != 0
    assert f"{name} must be a non-negative integer" in completed.stderr
    assert not marker.exists()


def _serve_models(ids: list[str]):
    """A minimal gateway exposing /readyz and /v1/models on an ephemeral port."""
    import json as _json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    payload = _json.dumps(
        {"object": "list", "data": [{"id": model_id} for model_id in ids]}
    ).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - http.server API
            body = b'{"status":"ready"}' if self.path == "/readyz" else payload
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802 - http.server API
            length = int(self.headers.get("Content-Length", "0"))
            request_body = self.rfile.read(length)
            self.server.chat_bodies.append(_json.loads(request_body))
            body = _json.dumps(
                {
                    "id": "chatcmpl-preflight",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "OK"},
                            "finish_reason": "length",
                        }
                    ],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    server.chat_bodies = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_model_preflight_compares_ids_exactly(tmp_path):
    """MODEL must never be interpolated into the match pattern.

    `qwen3.32b` as a regex matches `qwen3X32b`, which would silently benchmark
    the wrong deployment.
    """
    server = _serve_models(["qwen3X32b"])
    try:
        port = server.server_address[1]
        completed, marker = _run_script(
            tmp_path, {"MODEL": "qwen3.32b"}, port=port
        )
        assert completed.returncode != 0
        assert "is not served" in completed.stderr
        assert not marker.exists()
    finally:
        server.shutdown()


def test_model_preflight_accepts_the_exact_id(tmp_path):
    server = _serve_models(["other-model", "qwen3-32b"])
    try:
        port = server.server_address[1]
        completed, marker = _run_script(
            tmp_path,
            {"MODEL": "qwen3-32b", "BENCH_LIMIT": "1", "RESULTS_DIR": str(tmp_path / "r")},
            port=port,
        )
        assert completed.returncode == 0, completed.stderr
        assert "serving qwen3-32b" in completed.stdout
        assert server.chat_bodies == [
            {
                "model": "qwen3-32b",
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "max_tokens": 1,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": True},
            }
        ]
        assert marker.exists(), "the benchmark should have been launched"
    finally:
        server.shutdown()


def test_model_preflight_rejects_a_prefix_of_a_served_id(tmp_path):
    server = _serve_models(["qwen3-32b-instruct"])
    try:
        port = server.server_address[1]
        completed, _ = _run_script(tmp_path, {"MODEL": "qwen3-32b"}, port=port)
        assert completed.returncode != 0
        assert "is not served" in completed.stderr
    finally:
        server.shutdown()
