"""Focused artifact-graph checks for ``kairyu validate``."""

from __future__ import annotations

import json

import pytest

from kairyu.deploy import validation as validation_module
from kairyu.deploy.validation import validate_deployment
from kairyu.engine import registry as registry_module
from kairyu.entrypoints.server.settings import ServerSettings


def _write_tiny_model_checkpoint(
    path,
    *,
    extra_missing_shard=False,
    fp8_per_tensor=False,
    quant_scheme=None,
    raw_config=None,
):
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    from kairyu.engine.core.quant_config import detect_quantization
    from kairyu.models.config import parse_model_config
    from kairyu.models.loader import build_model
    from kairyu.quant.linear import linear_factory

    raw_config = dict(
        raw_config
        or {
            "architectures": ["Qwen3ForCausalLM"],
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "intermediate_size": 32,
            "vocab_size": 64,
            "tie_word_embeddings": True,
        }
    )
    if quant_scheme is not None:
        from tests.quant_checkpoint_fixtures import QUANT_CONFIGS

        raw_config["quantization_config"] = QUANT_CONFIGS[quant_scheme]
    if fp8_per_tensor:
        if quant_scheme is not None:
            raise ValueError("fp8_per_tensor and quant_scheme are mutually exclusive")
        raw_config["quantization_config"] = {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
        }
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(raw_config),
        encoding="utf-8",
    )
    model_config = parse_model_config(raw_config)
    model = build_model(
        model_config,
        linear_factory=linear_factory(detect_quantization(raw_config)),
    )
    state = {
        name: (
            tensor.new_ones((1,))
            if fp8_per_tensor and name.endswith(".weight_scale")
            else tensor.contiguous()
        )
        for name, tensor in model.state_dict().items()
        if not (
            name == "lm_head.weight"
            and model_config.tie_word_embeddings
        )
    }
    shard = "model.safetensors"
    save_file(state, path / shard)
    weight_map = {name: shard for name in state}
    if extra_missing_shard:
        weight_map["unused.tensor"] = "missing-extra.safetensors"
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )
    tokenizer = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "token": 1},
            unk_token="[UNK]",
        )
    )
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "chat_template": (
                    "{% for message in messages %}"
                    "{{ message['role'] ~ ': ' ~ message['content'] ~ '\\n' }}"
                    "{% endfor %}"
                )
            }
        ),
        encoding="utf-8",
    )


def test_validation_does_not_resolve_or_render_tenant_credentials(
    tmp_path,
    monkeypatch,
):
    secret = "tenant-secret-that-must-not-appear"
    deployment = tmp_path / "deploy.yaml"
    deployment.write_text(
        f"""
server:
  api_keys_env: UNREAD_API_KEYS
engines:
  local: {{backend: mock}}
tenants:
  key_tenants:
    {secret}: team
  limits:
    team: {{requests_per_minute: 60}}
""",
        encoding="utf-8",
    )

    def unexpected_resolution(_settings):
        raise AssertionError("offline validation must not resolve credentials")

    monkeypatch.setattr(
        ServerSettings,
        "resolve_api_keys",
        unexpected_resolution,
    )

    report = validate_deployment(deployment)

    assert report.valid
    assert secret not in report.render_text()


def test_validation_redacts_tenant_key_from_invalid_value_location(tmp_path):
    secret = "tenant-secret-that-must-not-appear"
    deployment = tmp_path / "deploy.yaml"
    deployment.write_text(
        f"""
engines:
  local: {{backend: mock}}
tenants:
  key_tenants:
    {secret}: 7
""",
        encoding="utf-8",
    )

    report = validate_deployment(deployment)

    assert not report.valid
    rendered = report.render_text()
    assert secret not in rendered
    assert "tenants.key_tenants.<redacted>" in rendered


def test_validation_redacts_tenant_key_from_duplicate_key_location(tmp_path):
    secret = "tenant.secret.with.dots"
    deployment = tmp_path / "deploy.yaml"
    deployment.write_text(
        f"""
engines:
  local: {{backend: mock}}
tenants:
  key_tenants:
    {secret}:
      repeated: first
      repeated: second
""",
        encoding="utf-8",
    )

    rendered = validate_deployment(deployment).render_text()

    assert secret not in rendered
    assert "tenants.key_tenants.<redacted>" in rendered


def test_validation_escapes_control_characters_in_field_locations(tmp_path):
    deployment = tmp_path / "deploy.yaml"
    deployment.write_text(
        """
engines:
  "evil\\nFAKE ERROR\\u001b[31m": {backend: unknown}
""",
        encoding="utf-8",
    )

    rendered = validate_deployment(deployment).render_text()

    assert "\x1b" not in rendered
    assert "\nFAKE ERROR" not in rendered
    assert r"evil\nFAKE ERROR\x1b[31m" in rendered


def test_validation_reports_root_yaml_recursion_as_invalid(tmp_path, monkeypatch):
    deployment = tmp_path / "deploy.yaml"
    deployment.write_text("engines: {}\n", encoding="utf-8")

    def recursive_parse(*_args, **_kwargs):
        raise RecursionError

    monkeypatch.setattr(validation_module.yaml, "load", recursive_parse)

    report = validate_deployment(deployment)

    assert not report.valid
    assert report.findings[0].code == "schema.invalid_yaml"


def test_validation_reports_linked_yaml_recursion_as_invalid(tmp_path, monkeypatch):
    orchestrator = tmp_path / "orchestrator.yaml"
    orchestrator.write_text("workers: []\n", encoding="utf-8")
    deployment = tmp_path / "deploy.yaml"
    deployment.write_text(
        """
engines:
  local: {backend: mock}
orchestrator: {spec: orchestrator.yaml}
""",
        encoding="utf-8",
    )

    def recursive_parse(*_args, **_kwargs):
        raise RecursionError

    monkeypatch.setattr(validation_module, "load_spec", recursive_parse)

    report = validate_deployment(deployment)

    assert not report.valid
    assert any(
        finding.artifact.endswith("orchestrator.yaml")
        and finding.code == "schema.invalid_yaml"
        for finding in report.findings
    )


def test_validation_reports_json_recursion_as_invalid(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    deployment = tmp_path / "deploy.yaml"
    deployment.write_text(
        f"""
engines:
  local:
    backend: kairyu
    options:
      model_path: {model}
""",
        encoding="utf-8",
    )

    def recursive_parse(*_args, **_kwargs):
        raise RecursionError

    monkeypatch.setattr(validation_module.json, "loads", recursive_parse)

    report = validate_deployment(deployment)

    assert not report.valid
    assert any(
        finding.artifact.endswith("config.json")
        and finding.code == "schema.invalid_json"
        for finding in report.findings
    )


def test_validation_reuses_role_graph_rules(tmp_path):
    orchestrator = tmp_path / "cycle.yaml"
    orchestrator.write_text(
        """
workers:
  - {name: worker, backend: mock}
roles:
  - name: first
    worker: worker
    prompt: first
    depends_on: [second]
  - name: second
    worker: worker
    prompt: second
    depends_on: [first]
""",
        encoding="utf-8",
    )
    deployment = tmp_path / "deploy.yaml"
    deployment.write_text(
        """
engines:
  local: {backend: mock}
orchestrator: {spec: cycle.yaml}
""",
        encoding="utf-8",
    )

    report = validate_deployment(deployment)

    assert not report.valid
    assert any(
        finding.code == "schema.invalid_role_graph"
        and "cycle" in finding.message
        for finding in report.findings
    )


def test_validation_resolves_links_from_the_config_symlink_location(tmp_path):
    release = tmp_path / "release"
    entry = tmp_path / "entry"
    release.mkdir()
    entry.mkdir()
    target = release / "deploy.yaml"
    target.write_text(
        """
engines:
  local: {backend: mock}
orchestrator: {spec: auto.yaml}
legacy_chat_models: [kairyu-auto]
""",
        encoding="utf-8",
    )
    (entry / "auto.yaml").write_text(
        "workers:\n  - {name: worker, backend: mock}\n",
        encoding="utf-8",
    )
    link = entry / "deploy.yaml"
    link.symlink_to(target)

    report = validate_deployment(link)

    assert report.valid


def test_validation_reports_orchestrator_symlink_loop(tmp_path):
    deployment = tmp_path / "deploy.yaml"
    deployment.write_text(
        """
engines:
  local: {backend: mock}
orchestrators:
  loop: {spec: loop.yaml}
  nul: {spec: "bad\\0path.yaml"}
""",
        encoding="utf-8",
    )
    loop = tmp_path / "loop.yaml"
    loop.symlink_to(loop)

    report = validate_deployment(deployment)

    assert sum(
        finding.code == "filesystem.missing_reference"
        for finding in report.findings
    ) == 2


def test_validation_rejects_unknown_backend_adapter(tmp_path):
    deployment = tmp_path / "deploy.yaml"
    deployment.write_text(
        """
engines:
  local: {backend: not-registered}
""",
        encoding="utf-8",
    )

    report = validate_deployment(deployment)

    assert any(
        finding.code == "schema.unknown_backend"
        and finding.field == "engines.local.backend"
        for finding in report.findings
    )


def test_validation_checks_model_metadata_without_loading_weights(
    tmp_path,
    monkeypatch,
):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "intermediate_size": 31,
                "vocab_size": 64,
            }
        ),
        encoding="utf-8",
    )
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": "missing.safetensors"}}),
        encoding="utf-8",
    )
    deployment = tmp_path / "deploy.yaml"
    deployment.write_text(
        f"""
engines:
  local:
    backend: kairyu
    options:
      model_path: {model}
      tensor_parallel_size: 2
""",
        encoding="utf-8",
    )

    def unexpected_model_load(*_args, **_kwargs):
        raise AssertionError("validation must not load model weights")

    monkeypatch.setattr(
        "kairyu.models.loader.load_model",
        unexpected_model_load,
    )

    report = validate_deployment(deployment)

    assert not report.valid
    assert any(
        finding.code == "filesystem.missing_shard"
        for finding in report.findings
    )
    assert any(
        finding.code == "schema.incompatible_tensor_parallel_size"
        for finding in report.findings
    )


def test_validation_reports_malformed_model_metadata_without_raising(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "hidden_size": "16",
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "intermediate_size": 32,
                "vocab_size": 64,
                "rope_scaling": [1],
            }
        ),
        encoding="utf-8",
    )
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": "empty.safetensors"}}),
        encoding="utf-8",
    )
    (model / "empty.safetensors").write_bytes(b"")
    deployment = tmp_path / "deploy.yaml"
    deployment.write_text(
        f"""
engines:
  local:
    backend: kairyu
    options:
      model_path: {model}
""",
        encoding="utf-8",
    )

    report = validate_deployment(deployment)

    assert not report.valid
    assert any(
        finding.code == "schema.incompatible_model"
        for finding in report.findings
    )


def test_validation_rejects_invalid_local_model_artifacts(tmp_path):
    from safetensors.torch import save_file

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "intermediate_size": 32,
                "vocab_size": 64,
            }
        ),
        encoding="utf-8",
    )
    (model / "generation_config.json").write_text(
        '{"eos_token_id": {}}',
        encoding="utf-8",
    )
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": "empty.safetensors"}}),
        encoding="utf-8",
    )
    save_file({}, model / "empty.safetensors")
    deployment = tmp_path / "deploy.yaml"
    deployment.write_text(
        f"""
engines:
  local:
    backend: kairyu
    options:
      model_path: {model}
""",
        encoding="utf-8",
    )

    report = validate_deployment(deployment)

    codes = {finding.code for finding in report.findings}
    assert {
        "schema.invalid_generation_config",
        "schema.invalid_checkpoint",
        "schema.invalid_tokenizer",
    } <= codes


def test_validation_rejects_missing_shard_for_unconsumed_index_entry(tmp_path):
    model = tmp_path / "model"
    _write_tiny_model_checkpoint(model, extra_missing_shard=True)
    deployment = tmp_path / "deploy.yaml"
    deployment.write_text(
        f"""
engines:
  local:
    backend: kairyu
    options:
      model_path: {model}
""",
        encoding="utf-8",
    )

    report = validate_deployment(deployment)

    assert not report.valid
    assert any(
        finding.code == "schema.invalid_checkpoint"
        for finding in report.findings
    )


@pytest.mark.parametrize(
    "quant_scheme",
    [
        pytest.param(None, id="dense"),
        pytest.param("fp8", id="fp8"),
        pytest.param("int8", id="int8"),
        pytest.param("awq", id="awq"),
        pytest.param("gptq", id="gptq"),
        pytest.param("nvfp4", id="nvfp4"),
    ],
)
def test_validation_accepts_runtime_checkpoint_member_contract(
    tmp_path,
    quant_scheme,
):
    model = tmp_path / (quant_scheme or "dense")
    _write_tiny_model_checkpoint(model, quant_scheme=quant_scheme)
    deployment = tmp_path / "deploy.yaml"
    deployment.write_text(
        f"""
engines:
  local:
    backend: kairyu
    options:
      model_path: {model}
""",
        encoding="utf-8",
    )

    report = validate_deployment(deployment)

    assert report.valid, report.render_text()


@pytest.mark.parametrize(
    ("mode", "expected_valid"),
    [("auto", False), ("vllm", False), ("none", True)],
)
def test_generation_config_mode_controls_deployment_sidecar_validation(
    tmp_path,
    mode,
    expected_valid,
):
    model = tmp_path / "model"
    _write_tiny_model_checkpoint(model)
    (model / "generation_config.json").write_text("{bad", encoding="utf-8")
    deployment = tmp_path / "deploy.yaml"
    deployment.write_text(
        f"""
engines:
  local:
    backend: kairyu
    options:
      model_path: {model}
      generation_config: {mode}
""",
        encoding="utf-8",
    )

    report = validate_deployment(deployment)

    generation_codes = {
        finding.code
        for finding in report.findings
        if finding.artifact.endswith("generation_config.json")
    }
    assert report.valid is expected_valid
    assert generation_codes == (
        {"schema.invalid_json"} if not expected_valid else set()
    )


def test_validation_accepts_deepseek_router_correction_buffer(tmp_path):
    model = tmp_path / "deepseek"
    _write_tiny_model_checkpoint(
        model,
        raw_config={
            "architectures": ["DeepseekV3ForCausalLM"],
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "intermediate_size": 64,
            "vocab_size": 64,
            "kv_lora_rank": 8,
            "q_lora_rank": None,
            "qk_nope_head_dim": 4,
            "qk_rope_head_dim": 4,
            "v_head_dim": 6,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 16,
            "n_group": 2,
            "topk_group": 1,
            "n_shared_experts": 1,
            "first_k_dense_replace": 0,
        },
    )
    index = json.loads(
        (model / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    correction = "model.layers.0.mlp.gate.e_score_correction_bias"
    assert correction in index["weight_map"]
    deployment = tmp_path / "deploy.yaml"
    deployment.write_text(
        f"""
engines:
  local:
    backend: kairyu
    options:
      model_path: {model}
""",
        encoding="utf-8",
    )

    report = validate_deployment(deployment)

    assert report.valid, report.render_text()


def test_validation_matches_runtime_fp8_scale_shape_contract(tmp_path):
    model = tmp_path / "model"
    _write_tiny_model_checkpoint(model, fp8_per_tensor=True)
    deployment = tmp_path / "deploy.yaml"
    deployment.write_text(
        f"""
engines:
  local:
    backend: kairyu
    options:
      model_path: {model}
""",
        encoding="utf-8",
    )

    assert validate_deployment(deployment).valid

    deployment.write_text(
        f"""
engines:
  local:
    backend: kairyu
    options:
      model_path: {model}
      tensor_parallel_size: 2
""",
        encoding="utf-8",
    )
    report = validate_deployment(deployment)

    assert any(
        finding.code == "schema.invalid_checkpoint"
        for finding in report.findings
    )


def test_validation_checks_each_typed_tokenizer_reference(tmp_path):
    model = tmp_path / "model"
    _write_tiny_model_checkpoint(model)
    deployment = tmp_path / "deploy.yaml"
    deployment.write_text(
        f"""
engines:
  default-tokenizer:
    backend: kairyu
    options:
      model_path: {model}
  literal-none-tokenizer:
    backend: kairyu
    options:
      model_path: {model}
      tokenizer: "None"
""",
        encoding="utf-8",
    )

    report = validate_deployment(deployment)

    assert any(
        finding.field == "engines.literal-none-tokenizer.options.tokenizer"
        for finding in report.findings
    )


def test_validation_requires_explicit_chat_policy_for_real_text_model(tmp_path):
    deployment = tmp_path / "deploy.yaml"
    deployment.write_text(
        """
engines:
  remote:
    backend: openai
    options:
      base_url: http://replica.test/v1
      model: upstream-model
      upstream: openai
""",
        encoding="utf-8",
    )

    report = validate_deployment(deployment)

    assert not report.valid
    assert [finding.code for finding in report.findings] == [
        "schema.missing_legacy_chat_policy"
    ]
    assert report.findings[0].field == "legacy_chat_models"


def test_validation_accepts_local_tokenizer_chat_template(tmp_path):
    model = tmp_path / "model"
    _write_tiny_model_checkpoint(model)
    deployment = tmp_path / "deploy.yaml"
    deployment.write_text(
        f"""
engines:
  local:
    backend: kairyu
    options:
      model_path: {model}
""",
        encoding="utf-8",
    )

    report = validate_deployment(deployment)

    assert report.valid, report.render_text()


def test_validation_reports_invalid_auto_loaded_chat_template(tmp_path):
    model = tmp_path / "model"
    _write_tiny_model_checkpoint(model)
    config_path = model / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["chat_template"] = "{% if"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    deployment = tmp_path / "deploy.yaml"
    deployment.write_text(
        f"""
engines:
  local:
    backend: kairyu
    options:
      model_path: {model}
""",
        encoding="utf-8",
    )

    report = validate_deployment(deployment)

    assert not report.valid
    assert [finding.code for finding in report.findings] == [
        "schema.invalid_chat_policy"
    ]


def test_validation_accepts_explicit_legacy_chat_policy_without_warning(
    tmp_path,
    caplog,
):
    deployment = tmp_path / "deploy.yaml"
    deployment.write_text(
        """
engines:
  remote:
    backend: openai
    options:
      base_url: http://replica.test/v1
      model: upstream-model
      upstream: openai
legacy_chat_models: [remote]
""",
        encoding="utf-8",
    )

    report = validate_deployment(deployment)

    assert report.valid, report.render_text()
    assert not caplog.records


@pytest.mark.parametrize(
    ("source", "contents", "expected_code"),
    [
        ("missing.jinja", None, "filesystem.missing_reference"),
        ("broken.jinja", "{% if", "schema.invalid_chat_template"),
    ],
)
def test_validation_preserves_precise_explicit_template_finding(
    tmp_path,
    source,
    contents,
    expected_code,
):
    if contents is not None:
        (tmp_path / source).write_text(contents, encoding="utf-8")
    deployment = tmp_path / "deploy.yaml"
    deployment.write_text(
        f"""
engines:
  remote:
    backend: openai
    options:
      base_url: http://replica.test/v1
      model: upstream-model
      upstream: openai
chat_templates:
  remote: {source}
""",
        encoding="utf-8",
    )

    report = validate_deployment(deployment)

    assert [finding.code for finding in report.findings] == [expected_code]


def test_validation_rejects_empty_explicit_chat_template(tmp_path):
    deployment = tmp_path / "deploy.yaml"
    deployment.write_text(
        """
engines:
  remote:
    backend: openai
    options:
      base_url: http://replica.test/v1
      model: upstream-model
chat_templates:
  remote: "   "
""",
        encoding="utf-8",
    )

    report = validate_deployment(deployment)

    assert [finding.code for finding in report.findings] == [
        "schema.invalid_chat_template"
    ]


def test_validation_respects_registered_native_name_override(
    tmp_path,
    monkeypatch,
):
    def replacement_factory(**_options):
        return object()

    monkeypatch.setitem(
        registry_module._FACTORIES,
        "kairyu",
        replacement_factory,
    )
    deployment = tmp_path / "deploy.yaml"
    deployment.write_text(
        """
engines:
  replacement:
    backend: kairyu
    options:
      model_path: not-a-local-model
legacy_chat_models: [replacement]
""",
        encoding="utf-8",
    )

    assert validate_deployment(deployment).valid
