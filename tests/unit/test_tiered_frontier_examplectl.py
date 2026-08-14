from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from kairyu.deploy.spec import load_deployment_spec
from kairyu.dsl.loader import load_spec
from kairyu.engine.config_validation import validate_backend_options

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
        "deepseek-v4-flash-0731-thinking",
    ):
        assert attribution in browser


def test_tiered_example_allocates_four_qwen_replicas_and_one_deepseek_tp4() -> None:
    spec = json.loads((EXAMPLE / "example.json").read_text())
    compose = yaml.safe_load((EXAMPLE / "compose.yaml").read_text())

    assert spec["hardware"] == {
        "gpu_count": 8,
        "product": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "minimum_compute_capability": 12.0,
        "minimum_vram_mib": 90000,
    }
    assert spec["deepseek_l1_loopback_port"] == 8005
    assert spec["verification"]["serving"]["auto_max_combined_max_tokens"] == 4096
    assert spec["allocation"] == {
        "tier1": {
            "model": "qwen3.8-27b",
            "gpu_ids": [0, 1, 2, 3],
            "replicas": 4,
            "tensor_parallel_size": 1,
        },
        "tier2": {
            "model": "deepseek-v4-flash-0731",
            "gpu_ids": [4, 5, 6, 7],
            "replicas": 1,
            "tensor_parallel_size": 4,
            "expert_parallel_size": 4,
        },
    }
    assert set(compose["services"]) == {
        "qwen-0",
        "qwen-1",
        "qwen-2",
        "qwen-3",
        "deepseek",
        "kairyu",
        "chat-ui",
    }
    assert spec["vllm"]["qwen"]["release"] == "v0.23.0"
    assert {
        compose["services"][service]["image"]
        for service in ("qwen-0", "qwen-1", "qwen-2", "qwen-3")
    } == {f"${{QWEN_VLLM_IMAGE:-{spec['vllm']['qwen']['image']}}}"}
    assert compose["services"]["deepseek"]["image"] == (
        f"${{DEEPSEEK_VLLM_IMAGE:-{spec['vllm']['deepseek']['image']}}}"
    )
    for index in range(4):
        service = compose["services"][f"qwen-{index}"]
        devices = service["deploy"]["resources"]["reservations"]["devices"][0]
        assert devices["device_ids"] == [str(index)]
        assert "--tensor-parallel-size" not in service["command"]
        assert "--language-model-only" not in service["command"]
        assert json.loads(_option(service["command"], "--mm-processor-kwargs")) == {
            "min_pixels": 65_536,
            "max_pixels": 2_097_152,
        }
        assert _option(service["command"], "--limit-mm-per-prompt.image") == "1"
        assert _option(service["command"], "--limit-mm-per-prompt.video") == "0"
        assert json.loads(
            _option(service["command"], "--default-chat-template-kwargs")
        ) == {"enable_thinking": False}
        assert _option(service["command"], "--max-num-seqs") == "32"
        assert _option(service["command"], "--max-num-batched-tokens") == "32768"
        assert _option(service["command"], "--kv-cache-dtype") == "fp8"
        assert json.loads(
            _option(service["command"], "--compilation-config")
        ) == {"cudagraph_mode": "PIECEWISE"}
        assert "--speculative-config" not in service["command"]
        assert service["volumes"][-2]["target"] == "/root/.cache"
        assert service["volumes"][-1] == (
            "../qwen3.8-27b-1gpu/chat_template.jinja:"
            "/etc/kairyu/qwen3.8-chat.jinja:ro"
        )
        assert service["environment"] | {
            "XDG_CACHE_HOME": "/root/.cache",
            "TORCHINDUCTOR_CACHE_DIR": "/root/.cache/torchinductor",
            "TRITON_CACHE_DIR": "/root/.cache/triton",
            "TILELANG_CACHE_DIR": "/root/.cache/tilelang",
            "TILELANG_TMP_DIR": "/root/.cache/tilelang/tmp",
        } == service["environment"]
    deepseek = compose["services"]["deepseek"]
    devices = deepseek["deploy"]["resources"]["reservations"]["devices"][0]
    assert devices["device_ids"] == ["4", "5", "6", "7"]
    assert _option(deepseek["command"], "--tensor-parallel-size") == "4"
    assert deepseek["ports"] == [
        "127.0.0.1:${DEEPSEEK_L1_PORT:-8005}:8000"
    ]
    assert "--enable-expert-parallel" in deepseek["command"]
    assert "--enable-auto-tool-choice" in deepseek["command"]
    assert _option(deepseek["command"], "--tool-call-parser") == "deepseek_v4"
    assert _option(deepseek["command"], "--max-num-batched-tokens") == "16384"
    assert deepseek["volumes"][1]["target"] == "/root/.cache"
    assert deepseek["environment"] | {
        "XDG_CACHE_HOME": "/root/.cache",
        "TORCHINDUCTOR_CACHE_DIR": "/root/.cache/torchinductor",
        "TRITON_CACHE_DIR": "/root/.cache/triton",
        "TILELANG_CACHE_DIR": "/root/.cache/tilelang",
        "TILELANG_TMP_DIR": "/root/.cache/tilelang/tmp",
    } == deepseek["environment"]
    assert json.loads(_option(deepseek["command"], "--speculative-config")) == {
        "method": "dspark",
        "num_speculative_tokens": 5,
        "draft_sample_method": "greedy",
    }
    assert json.loads(_option(deepseek["command"], "--compilation-config")) == {
        "cudagraph_mode": "FULL_AND_PIECEWISE",
        "custom_ops": ["all"],
    }
    assert compose["services"]["kairyu"]["build"]["args"] == {
        "KAIRYU_VISION": "1",
        "KAIRYU_EMBEDDINGS": "1",
        "KAIRYU_EMBEDDING_MODEL_REPOSITORY": EMBEDDING_MODEL_REPOSITORY,
        "KAIRYU_EMBEDDING_MODEL_REVISION": EMBEDDING_MODEL_REVISION,
        "KAIRYU_EMBEDDING_MODEL_SHA256": EMBEDDING_MODEL_SHA256,
        "KAIRYU_EMBEDDING_PROVENANCE_SHA256": EMBEDDING_PROVENANCE_SHA256,
    }
    assert spec["embedding"] == {
        "served_name": "embed-small",
        "backend": "fastembed",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "model_path": "/opt/kairyu/models/all-MiniLM-L6-v2",
        "repository": EMBEDDING_MODEL_REPOSITORY,
        "revision": EMBEDDING_MODEL_REVISION,
        "model_sha256": EMBEDDING_MODEL_SHA256,
        "provenance_sha256": EMBEDDING_PROVENANCE_SHA256,
        "dimensions": 384,
        "batch_size": 64,
        "threads": 2,
        "max_concurrency": 2,
    }
    assert compose["services"]["kairyu"]["ports"] == [
        "${API_BIND_ADDRESS:-0.0.0.0}:${API_PORT:-8003}:8000"
    ]


def test_tiered_gateway_owns_l2_pools_templates_and_orchestrators() -> None:
    raw = (EXAMPLE / "kairyu.yaml").read_text()
    deployment = load_deployment_spec(raw, resolve_credentials=False)

    assert set(deployment.pools) == {
        "qwen3.8-27b",
        "deepseek-v4-flash-0731",
        "deepseek-v4-flash-0731-thinking",
    }
    qwen = deployment.pools["qwen3.8-27b"]
    assert len(qwen.replicas) == 4
    assert qwen.prefix_index is True
    assert qwen.queue_depth_threshold == 0
    for replica in qwen.replicas:
        validate_backend_options(replica.backend, replica.options)
        assert replica.options["tensor_parallel_size"] == 1
        assert replica.options["upstream"] == "vllm"
        assert replica.options["capabilities"] == {
            "allow_prompt_kinds": ["multimodal"],
            "allow_chat_template_kwargs": ["enable_thinking"],
        }
        assert replica.options["image_input_policy"]["max_images"] == 1
        assert replica.options["container_image_digest"] == (
            "sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f"
        )
    deepseek = deployment.pools["deepseek-v4-flash-0731"]
    assert len(deepseek.replicas) == 1
    validate_backend_options(deepseek.replicas[0].backend, deepseek.replicas[0].options)
    assert deepseek.replicas[0].options["tensor_parallel_size"] == 4
    assert deepseek.replicas[0].options["expert_parallel_size"] == 4
    assert deepseek.replicas[0].options["dspark_enabled"] is True
    assert deepseek.replicas[0].options["container_image_digest"] == (
        "sha256:99756b54424a4697f69476b29aa02fb7f8112aaa74fa8203a7bf8a0bae4ca6f1"
    )
    assert "completion_reasoning_end_tag" not in deepseek.replicas[0].options
    thinking = deployment.pools["deepseek-v4-flash-0731-thinking"]
    assert thinking.replicas[0].options["completion_reasoning_end_tag"] == "</think>"
    assert set(deployment.orchestrators) == {
        "kairyu-auto-max",
    }
    assert list(deployment.embeddings) == ["embed-small"]
    embedding = deployment.embeddings["embed-small"]
    assert embedding.backend == "fastembed"
    assert embedding.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert embedding.model_path == "/opt/kairyu/models/all-MiniLM-L6-v2"
    assert embedding.revision == EMBEDDING_MODEL_REVISION
    assert embedding.model_sha256 == EMBEDDING_MODEL_SHA256
    assert embedding.provenance_sha256 == EMBEDDING_PROVENANCE_SHA256
    assert embedding.dimensions == 384
    assert embedding.batch_size == 64
    assert embedding.threads == 2
    assert embedding.max_concurrency == 2
    assert deployment.public_models == frozenset(
        {"kairyu-auto-max", "embed-small"}
    )
    assert deployment.chat_templates == {
        "deepseek-v4-flash-0731": "/etc/kairyu/deepseek-v4-0731.jinja",
        "deepseek-v4-flash-0731-thinking": "/etc/kairyu/deepseek-thinking.jinja",
    }
    assert deployment.legacy_chat_models == frozenset({"qwen3.8-27b"})
    assert deployment.server.max_chat_body_bytes == 16_777_216


def test_tiered_private_reasoning_prompt_converges_without_requesting_a_transcript() -> None:
    template = (EXAMPLE / "deepseek-thinking.jinja").read_text()

    assert "Reason privately and carefully" in template
    assert "converge promptly" in template
    assert "non-empty final answer immediately follows </think>" in template
    assert "entire deliberation process" not in template
    assert "Do not stop reasoning" not in template


def test_tiered_l2_pins_only_the_explicit_product_dag() -> None:
    config = json.loads((EXAMPLE / "example.json").read_text())
    maximum = load_spec(EXAMPLE / "auto-max.yaml")

    assert [worker.name for worker in maximum.workers] == [
        "tier1",
        "tier2",
        "publisher",
    ]
    assert maximum.workers[0].engine_ref == "qwen3.8-27b"
    assert maximum.workers[1].engine_ref == "deepseek-v4-flash-0731-thinking"
    assert maximum.workers[2].engine_ref == "qwen3.8-27b"
    assert maximum.router.kind == "calibrated"
    assert maximum.router.target_mode == "auto-max"
    assert maximum.moa_samples == 0
    assert maximum.internal_max_tokens == 2048
    assert maximum.expose_intermediate_outputs is True
    assert config["orchestration"]["internal_max_output_tokens"] == 2048
    assert config["orchestration"]["product_policy"] == "verifier-gated-role-dag"
    assert config["orchestration"]["product_normal_calls"] == 7
    assert config["orchestration"]["product_max_calls"] == 11
    assert config["orchestration"]["product_max_refinements"] == 2
    assert maximum.budget.max_steps == 11
    assert maximum.budget.max_refine_depth == 2
    assert [role.name for role in maximum.roles] == [
        "planner",
        "proposal_a",
        "proposal_b",
        "proposal_c",
        "draft_synthesis",
        "verifier",
        "publisher",
    ]
    verifier = next(role for role in maximum.roles if role.name == "verifier")
    publisher = next(role for role in maximum.roles if role.name == "publisher")
    assert verifier.verifies == "draft_synthesis"
    assert publisher.worker == "publisher"
    assert publisher.depends_on == ("draft_synthesis", "verifier")
    assert sorted(path.name for path in EXAMPLE.glob("auto*.yaml")) == ["auto-max.yaml"]
    assert "base_url: http://kairyu:8000/v1" not in (EXAMPLE / "auto-max.yaml").read_text()


def test_tiered_chat_ui_calls_kairyu_l3() -> None:
    compose = yaml.safe_load((EXAMPLE / "compose.yaml").read_text())
    ui = compose["services"]["chat-ui"]
    assert ui["environment"]["OPENAI_API_BASE_URL"] == "http://kairyu:8000/v1"
    assert json.loads(ui["environment"]["DEFAULT_MODEL_PARAMS"]) == {
        "max_tokens": 32768,
        "stream_response": False,
    }
    assert ui["environment"] | {
        "DEFAULT_MODELS": "kairyu-auto-max",
        "OPENAI_API_CONFIGS": '{"0":{"model_ids":["kairyu-auto-max"]}}',
        "ENABLE_PERSISTENT_CONFIG": "false",
        "RESET_CONFIG_ON_START": "true",
        "ENABLE_SIGNUP": "false",
        "ENABLE_LOGIN_FORM": "false",
        "WEBUI_AUTH": "false",
        "ENABLE_EVALUATION_ARENA_MODELS": "false",
    } == ui["environment"]
    assert ui["ports"] == ["${CHAT_UI_BIND_ADDRESS:-0.0.0.0}:${CHAT_UI_PORT:-3000}:8080"]
    assert ui["depends_on"] == {"kairyu": {"condition": "service_healthy"}}
    assert compose["services"]["kairyu"]["depends_on"] == {
        "qwen-0": {"condition": "service_healthy"},
        "qwen-1": {"condition": "service_healthy"},
        "qwen-2": {"condition": "service_healthy"},
        "qwen-3": {"condition": "service_healthy"},
        "deepseek": {"condition": "service_healthy"},
    }
    policy_mounts = {
        volume for volume in compose["services"]["kairyu"]["volumes"]
        if isinstance(volume, str) and "/auto" in volume
    }
    assert policy_mounts == {"./auto-max.yaml:/etc/kairyu/auto-max.yaml:ro"}


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
                            "planner",
                            "proposal_a",
                            "proposal_b",
                            "proposal_c",
                            "draft_synthesis",
                            "verifier",
                            "publisher",
                        )
                    ],
                    "moa_samples": 0,
                    "budget": {"max_steps": 11, "max_refine_depth": 2},
                    "expose_intermediate_outputs": True,
                    "configured_engines": {
                        "tier1": {"model": "qwen3.8-27b"},
                        "tier2": {"model": "deepseek-v4-flash-0731-thinking"},
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
            {"model": "deepseek-v4-flash-0731", "prompt": "kairyu"},
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



def test_tiered_product_serving_requires_publisher_trace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    benchmark = _load(EXAMPLE / "verification.py", "tiered_example_product_verification")
    observed: list[tuple[str, dict]] = []

    def fake_serving(model, _run_dir, **kwargs):
        observed.append((model, kwargs))
        return 0

    monkeypatch.setattr(benchmark, "_serving", fake_serving)
    assert benchmark.serving_auto_max(tmp_path) == 0

    assert observed == [
        (
            "kairyu-auto-max",
            {
                "tensor_parallel": 4,
                "replicas": 1,
                "expected_route": "publisher",
                "expected_role": "publisher",
                "expected_kind": "generation",
                "warmup_requests": 4,
                "natural_completion": True,
            },
        )
    ]
