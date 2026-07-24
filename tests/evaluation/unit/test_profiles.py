from importlib import resources

import pytest
from pydantic import ValidationError

from kairyu.evaluation import profiles as profiles_module
from kairyu.evaluation.profiles import (
    ProfileDataError,
    load_profile_resource,
    load_profiles,
)


def _gpqa_profile_text() -> str:
    return (
        resources.files("kairyu.evaluation")
        .joinpath("resources", "profiles", "gpqa-diamond.yaml")
        .read_text(encoding="utf-8")
    )


def _install_profile_resource(
    tmp_path,
    monkeypatch,
    *,
    benchmark_id: str,
    text: str,
):
    package_root = tmp_path / "package"
    resource = package_root / "resources" / "profiles" / f"{benchmark_id}.yaml"
    resource.parent.mkdir(parents=True)
    resource.write_text(text, encoding="utf-8")
    monkeypatch.setattr(profiles_module, "files", lambda _package: package_root)
    return resource


def test_gpqa_profile_conversion_preserves_legacy_protocol_defaults():
    profile = load_profiles("gpqa-diamond")[0]

    assert profile.protocol.benchmark_version == "gpqa-diamond-evalscope-v1.8.1"
    assert profile.protocol.modalities == ("text",)
    assert profile.protocol.tools == ()
    assert profile.protocol.judge_model is None
    assert profile.protocol.adapter_configuration["choice_labels"] == ["A", "B", "C", "D"]
    assert profile.protocol.dependency_compatibility_patches == (
        "kairyu.evaluation.adapters.gpqa_v181@sha256:"
        "533999762a93d5a75694ea785212e131dadbf725b1862d9c5fc2f453c4774556",
    )


def test_hle_profile_conversion_uses_resource_specific_protocol_fields():
    resource = load_profile_resource("humanitys-last-exam")
    profiles = load_profiles("humanitys-last-exam")

    assert [profile.name for profile in resource.profiles] == [
        "smoke",
        "fugu-2026",
        "official-latest",
    ]
    assert all(lock.evalscope_wheel_sha256 is None for lock in resource.profiles)
    protocol = profiles[0].protocol
    assert protocol.benchmark_id == "humanitys-last-exam"
    assert protocol.benchmark_version == "humanitys-last-exam-cais-simple-evals-2026.07"
    assert protocol.modalities == ("text", "image")
    assert protocol.retries == 4
    assert protocol.judge_model == "gpt-5-mini"
    assert protocol.judge_prompt_version == "cais-simple-evals-hle-judge-2026.07"
    assert "choice_labels" not in protocol.adapter_configuration
    assert protocol.adapter_configuration["upstream_repository"] == (
        "https://github.com/centerforaisafety/simple-evals"
    )
    assert protocol.dependency_compatibility_patches == (
        "kairyu.evaluation.adapters.hle_official_2026@sha256:"
        f"{resource.profiles[0].compatibility_layer_sha256}",
    )


def test_profile_resource_rejects_duplicate_yaml_keys(tmp_path, monkeypatch):
    text = _gpqa_profile_text() + "\nbenchmark_id: gpqa-diamond\n"
    _install_profile_resource(
        tmp_path,
        monkeypatch,
        benchmark_id="gpqa-diamond",
        text=text,
    )

    with pytest.raises(ProfileDataError, match="strict YAML"):
        load_profile_resource("gpqa-diamond")


def test_profile_resource_rejects_duplicate_profile_names(tmp_path, monkeypatch):
    text = _gpqa_profile_text().replace("  - name: fugu-2026", "  - name: smoke", 1)
    _install_profile_resource(
        tmp_path,
        monkeypatch,
        benchmark_id="gpqa-diamond",
        text=text,
    )

    with pytest.raises(ProfileDataError, match="schema validation"):
        load_profile_resource("gpqa-diamond")


def test_gpqa_profile_resource_requires_evalscope_wheel_hash(tmp_path, monkeypatch):
    text = _gpqa_profile_text().replace(
        "    evalscope_wheel_sha256: "
        "0bbbd2302d038b2c3a95cad2c4b20b93cc4ca911dc7b0033060c551a22cbab8f\n",
        "",
        1,
    )
    _install_profile_resource(
        tmp_path,
        monkeypatch,
        benchmark_id="gpqa-diamond",
        text=text,
    )

    with pytest.raises(ProfileDataError, match="schema validation"):
        load_profile_resource("gpqa-diamond")


def test_profile_resource_rejects_mismatched_benchmark_id(tmp_path, monkeypatch):
    _install_profile_resource(
        tmp_path,
        monkeypatch,
        benchmark_id="humanitys-last-exam",
        text=_gpqa_profile_text(),
    )

    with pytest.raises(ProfileDataError, match="does not match its filename"):
        load_profile_resource("humanitys-last-exam")


def test_profile_resource_rejects_unsupported_schema_version(tmp_path, monkeypatch):
    text = _gpqa_profile_text().replace("schema_version: 1", "schema_version: 2", 1)
    _install_profile_resource(
        tmp_path,
        monkeypatch,
        benchmark_id="gpqa-diamond",
        text=text,
    )

    with pytest.raises(ProfileDataError, match="schema validation"):
        load_profile_resource("gpqa-diamond")


def test_profile_resource_rejects_unsafe_benchmark_id_before_lookup(monkeypatch):
    def fail_lookup(_package):
        raise AssertionError("unsafe benchmark ID reached resource lookup")

    monkeypatch.setattr(profiles_module, "files", fail_lookup)

    with pytest.raises(KeyError, match="invalid benchmark ID"):
        load_profile_resource("../gpqa-diamond")


def test_profile_lock_json_configuration_is_frozen():
    lock = load_profile_resource("humanitys-last-exam").profiles[0]

    with pytest.raises(TypeError):
        lock.adapter_configuration["request_concurrency"] = 8
    with pytest.raises(ValidationError):
        lock.retries = 0
