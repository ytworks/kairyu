"""Fail-closed measured crossover policy for #187."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from kairyu.engine.config_validation import validate_backend_options
from kairyu.engine.core.engine_service import (
    _engine_config_with_pinned_dram_profile,
)
from kairyu.engine.core.kv_pool import PagedKVPool
from kairyu.engine.core.kv_tier_policy import (
    POLICY_RULE,
    SCHEMA_VERSION,
    DramKVRuntimeIdentity,
    build_dram_kv_tier,
    engine_source_inventory,
    engine_source_sha256,
    load_crossover_policy,
    model_config_sha256,
    pcie_transport_identity,
)
from kairyu.engine.kairyu_backend import build_engine_loop
from kairyu.engine.zmq_backend import (
    EngineServiceError,
    ZmqEngineBackend,
    _dram_kv_tier_from_startup_frame,
    _validate_startup_dram_kv_tier_identity,
)

_DIGEST = "a" * 64
_MAX_NUM_BATCHED_TOKENS = 2048


def _attention_identity(
    backend: str = "flashinfer",
    *,
    prefill_version: str = "0.6.4",
    decode_version: str = "0.6.4",
) -> str:
    return json.dumps(
        {
            "requested": "auto",
            "resolved": backend,
            "source": "hw_profile",
            "components": {
                "prefill": backend,
                "decode": "flashinfer",
                "kv_mode": "paged-direct",
            },
            "component_versions": {
                "prefill": prefill_version,
                "decode": decode_version,
            },
            "component_builds": {
                "flashinfer-jit-cache": {
                    "installed": True,
                    "version": "0.6.4",
                    "record_sha256": "c" * 64,
                }
            },
            "architecture": {
                "arch": "cuda",
                "device_name": "test-gpu",
                "sm": 120,
                "kernel_tier": "full",
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    )


_ATTENTION_IDENTITY = _attention_identity()


def _identity() -> DramKVRuntimeIdentity:
    pool = PagedKVPool(
        num_layers=2,
        num_pages=8,
        page_size=16,
        num_kv_heads=2,
        head_dim=8,
    )
    return DramKVRuntimeIdentity.from_pool(
        pool,
        model_config_digest=_DIGEST,
        tensor_parallel_size=4,
        tensor_parallel_rank=0,
        attention_backend_identity=_ATTENTION_IDENTITY,
        max_num_batched_tokens=_MAX_NUM_BATCHED_TOKENS,
    )


def _samples() -> list[dict[str, object]]:
    rows = []
    ratios = {16: 1.10, 32: 0.94, 64: 0.80}
    for prefix_tokens, ratio in ratios.items():
        for repeat in range(9):
            observed_ratio = (
                1.01
                if prefix_tokens == 32 and repeat == 8
                else ratio
            )
            rows.append(
                {
                    "prefix_tokens": prefix_tokens,
                    "repeat": repeat,
                    "order": (
                        ["restore", "recompute"]
                        if repeat % 2 == 0
                        else ["recompute", "restore"]
                    ),
                    "restore_ns": round(1000 * observed_ratio),
                    "recompute_ns": 1000,
                    "restore_sha256": _DIGEST,
                    "recompute_sha256": _DIGEST,
                }
            )
    return rows


def _profile(identity: DramKVRuntimeIdentity) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "runtime_identity": dataclasses.asdict(identity),
        "runtime_fingerprint": identity.fingerprint,
        "samples": _samples(),
        "policy": {
            "rule": POLICY_RULE,
            "min_restore_tokens": 32,
        },
        "evidence_sha256": "b" * 64,
        "passed": True,
    }


def _write_profile(tmp_path, payload: dict[str, object]):
    path = tmp_path / "dram-kv-profile.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_profile_rederives_the_stable_crossover_from_raw_pairs(tmp_path) -> None:
    identity = _identity()
    path = _write_profile(tmp_path, _profile(identity))

    policy = load_crossover_policy(path, expected_identity=identity)

    assert policy.min_restore_tokens == 32
    assert dict(policy.median_ratios) == {16: 1.1, 32: 0.94, 64: 0.8}
    assert dict(policy.wins) == {16: 0, 32: 8, 64: 9}
    assert policy.evidence_sha256 == "b" * 64
    assert len(policy.profile_sha256) == 64


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda profile: profile.update(passed=False), "not a passing"),
        (
            lambda profile: profile["policy"].update(min_restore_tokens=16),
            "differs from the raw-sample",
        ),
        (
            lambda profile: profile["policy"].update(
                rule="median_only"
            ),
            "changed the binding decision rule",
        ),
        (
            lambda profile: profile["samples"][0].update(
                restore_sha256="b" * 64
            ),
            "digests must match",
        ),
        (
            lambda profile: profile["samples"].pop(),
            "exactly",
        ),
    ],
)
def test_profile_cannot_claim_a_verdict_its_raw_evidence_does_not_support(
    tmp_path,
    mutation,
    message: str,
) -> None:
    identity = _identity()
    profile = _profile(identity)
    mutation(profile)
    with pytest.raises(ValueError, match=message):
        load_crossover_policy(
            _write_profile(tmp_path, profile),
            expected_identity=identity,
        )


def test_profile_is_rejected_after_runtime_identity_drift(tmp_path) -> None:
    identity = _identity()
    changed = dataclasses.replace(identity, tensor_parallel_size=8)
    with pytest.raises(ValueError, match="does not match the current"):
        load_crossover_policy(
            _write_profile(tmp_path, _profile(identity)),
            expected_identity=changed,
        )


@pytest.mark.parametrize(
    "changed",
    [
        dataclasses.replace(
            _identity(),
            attention_backend_identity=_attention_identity(
                "flashattention3",
                prefill_version="3.0.0",
            ),
        ),
        dataclasses.replace(
            _identity(),
            attention_backend_identity=_attention_identity(
                "flashattention4",
                prefill_version="4.0.0b24",
            ),
        ),
        dataclasses.replace(
            _identity(),
            attention_backend_identity=_attention_identity(
                prefill_version="0.7.0",
                decode_version="0.7.0",
            ),
        ),
        dataclasses.replace(_identity(), max_num_batched_tokens=1024),
        dataclasses.replace(_identity(), cpu_model="different-cpu-model"),
        dataclasses.replace(_identity(), os_page_size=65536),
        dataclasses.replace(_identity(), engine_source_sha256="d" * 64),
    ],
    ids=(
        "flashattention3",
        "flashattention4",
        "package-version",
        "chunk-budget",
        "cpu-model",
        "os-page-size",
        "engine-source",
    ),
)
def test_profile_is_rejected_after_execution_policy_drift(
    tmp_path,
    changed: DramKVRuntimeIdentity,
) -> None:
    identity = _identity()
    with pytest.raises(ValueError, match="does not match the current"):
        load_crossover_policy(
            _write_profile(tmp_path, _profile(identity)),
            expected_identity=changed,
        )


def test_profile_rejects_a_forged_identity_fingerprint(tmp_path) -> None:
    identity = _identity()
    profile = _profile(identity)
    profile["runtime_identity"]["page_size"] = 32
    with pytest.raises(ValueError, match="fingerprint does not match"):
        load_crossover_policy(
            _write_profile(tmp_path, profile),
            expected_identity=identity,
        )


def test_profile_rejects_a_non_execution_attention_identity(tmp_path) -> None:
    identity = _identity()
    profile = _profile(identity)
    profile["runtime_identity"]["attention_backend_identity"] = "{}"
    with pytest.raises(ValueError, match="execution identity fields"):
        load_crossover_policy(
            _write_profile(tmp_path, profile),
            expected_identity=identity,
        )


def test_profile_requires_a_stable_margin_at_and_above_the_crossover(
    tmp_path,
) -> None:
    identity = _identity()
    profile = _profile(identity)
    for row in profile["samples"]:
        if row["prefix_tokens"] == 64:
            row["restore_ns"] = 990
    # Two losses leave only 7/9 wins even though the median remains faster.
    changed = 0
    for row in profile["samples"]:
        if row["prefix_tokens"] == 64 and changed < 2:
            row["restore_ns"] = 1010
            changed += 1
    with pytest.raises(ValueError, match="no stable measured crossover"):
        load_crossover_policy(
            _write_profile(tmp_path, profile),
            expected_identity=identity,
        )


def test_model_config_identity_is_canonical_not_whitespace_sensitive(
    tmp_path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "config.json").write_text(
        '{"hidden_size": 32, "architectures": ["Qwen3ForCausalLM"]}\n',
        encoding="utf-8",
    )
    (second / "config.json").write_text(
        '{\n  "architectures":["Qwen3ForCausalLM"],\n  "hidden_size":32\n}',
        encoding="utf-8",
    )
    assert model_config_sha256(first) == model_config_sha256(second)


def test_pcie_transport_identity_reads_the_stable_sysfs_cohort(tmp_path) -> None:
    device = tmp_path / "bus/pci/devices/0000:16:00.0"
    device.mkdir(parents=True)
    (device / "max_link_speed").write_text("32.0 GT/s PCIe\n", encoding="utf-8")
    (device / "max_link_width").write_text("16\n", encoding="utf-8")
    (device / "current_link_width").write_text("16\n", encoding="utf-8")
    (device / "numa_node").write_text("0\n", encoding="utf-8")

    assert pcie_transport_identity(
        "0000:16:00.0",
        sysfs_root=tmp_path,
    ) == ("32.0 GT/s PCIe", 16, 16, 0)


def test_engine_source_rollup_changes_with_any_runtime_module(
    tmp_path,
) -> None:
    package_root = tmp_path / "kairyu"
    model_source = package_root / "models/unlisted_model.py"
    attention_source = package_root / "engine/core/attention/new_backend.py"
    model_source.parent.mkdir(parents=True)
    attention_source.parent.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    model_source.write_text("value = 'model-v1'\n", encoding="utf-8")
    attention_source.write_text("value = 'attention-v1'\n", encoding="utf-8")

    inventory = engine_source_inventory(package_root)
    assert list(inventory) == sorted(inventory)
    assert "kairyu/models/unlisted_model.py" in inventory
    assert "kairyu/engine/core/attention/new_backend.py" in inventory
    before = engine_source_sha256(package_root)
    assert engine_source_sha256(package_root) == before
    model_source.write_text(
        "value = 'changed'\n",
        encoding="utf-8",
    )
    assert engine_source_sha256(package_root) != before


def test_engine_source_inventory_rejects_symlink_escape(tmp_path) -> None:
    package_root = tmp_path / "kairyu"
    package_root.mkdir()
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("value = 'outside'\n", encoding="utf-8")
    (package_root / "escaped.py").symlink_to(outside)

    with pytest.raises(ValueError, match="contains a symlink"):
        engine_source_inventory(package_root)


def test_runtime_binding_validates_before_allocating_a_rank_local_tier(
    tmp_path,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        '{"architectures":["Qwen3ForCausalLM"],"hidden_size":32}',
        encoding="utf-8",
    )
    pool = PagedKVPool(
        num_layers=2,
        num_pages=8,
        page_size=16,
        num_kv_heads=2,
        head_dim=8,
    )
    identity = DramKVRuntimeIdentity.from_pool(
        pool,
        model_config_digest=model_config_sha256(model_path),
        tensor_parallel_size=2,
        tensor_parallel_rank=0,
        attention_backend_identity=_ATTENTION_IDENTITY,
        max_num_batched_tokens=_MAX_NUM_BATCHED_TOKENS,
    )
    profile_path = _write_profile(tmp_path, _profile(identity))

    assert identity.host_memory_placement_policy == "not-applicable"
    assert identity.pcie_max_link_speed is None
    assert identity.pcie_max_link_width is None
    assert identity.pcie_current_link_width is None
    assert identity.local_numa_gpu_count is None
    assert identity.cpu_model
    assert identity.python_version
    assert identity.openssl_version

    with pytest.raises(ValueError, match="capacity_pages must be at least 2"):
        build_dram_kv_tier(
            pool,
            model_path=model_path,
            tensor_parallel_size=2,
            tensor_parallel_rank=0,
            capacity_pages=1,
            profile_path=profile_path,
            attention_backend_identity=_ATTENTION_IDENTITY,
            max_num_batched_tokens=_MAX_NUM_BATCHED_TOKENS,
        )

    rank0 = build_dram_kv_tier(
        pool,
        model_path=model_path,
        tensor_parallel_size=2,
        tensor_parallel_rank=0,
        capacity_pages=2,
        profile_path=profile_path,
        attention_backend_identity=_ATTENTION_IDENTITY,
        max_num_batched_tokens=_MAX_NUM_BATCHED_TOKENS,
    )
    rank1 = build_dram_kv_tier(
        pool,
        model_path=model_path,
        tensor_parallel_size=2,
        tensor_parallel_rank=1,
        capacity_pages=2,
        profile_path=profile_path,
        attention_backend_identity=_ATTENTION_IDENTITY,
        max_num_batched_tokens=_MAX_NUM_BATCHED_TOKENS,
    )

    assert rank0.policy.min_restore_tokens == 32
    assert rank0.tier.capacity_pages == 2
    assert rank0.handshake_identity == rank1.handshake_identity
    assert rank0.tier.fingerprint_digest != rank1.tier.fingerprint_digest


@pytest.mark.parametrize("backend", ["kairyu", "kairyu-proc"])
def test_native_config_requires_a_complete_dram_tier_pair(backend: str) -> None:
    with pytest.raises(ValueError, match="requires both"):
        validate_backend_options(
            backend,
            {
                "model_path": "/model",
                "dram_kv_tier_capacity_pages": 1,
            },
        )
    with pytest.raises(ValueError, match="requires both"):
        validate_backend_options(
            backend,
            {
                "model_path": "/model",
                "dram_kv_tier_profile": "/profile.json",
            },
        )


@pytest.mark.parametrize(
    "options",
    [
        {
            "dram_kv_tier_capacity_pages": 1,
            "dram_kv_tier_profile": "/profile.json",
        },
        {
            "model_path": "/model",
            "pd_separation": True,
            "dram_kv_tier_capacity_pages": 1,
            "dram_kv_tier_profile": "/profile.json",
        },
    ],
)
def test_native_config_rejects_unsupported_dram_tier_ownership(
    options,
) -> None:
    match = "real model_path" if "model_path" not in options else "P-D separation"
    with pytest.raises(ValueError, match=match):
        validate_backend_options("kairyu", options)


def test_direct_builder_rejects_incomplete_dram_tier_before_model_load() -> None:
    with pytest.raises(ValueError, match="requires both"):
        build_engine_loop(
            model_path="/missing-model",
            dram_kv_tier_capacity_pages=1,
        )


def test_process_backend_forwards_and_pins_the_complete_dram_tier_config(
    tmp_path,
) -> None:
    profile_path = tmp_path / "profile.json"
    profile_bytes = b'{"measured":"profile-v1"}'
    profile_path.write_bytes(profile_bytes)
    backend = ZmqEngineBackend(
        model_path="/model",
        dram_kv_tier_capacity_pages=8,
        dram_kv_tier_profile=profile_path,
    )

    assert backend._config["dram_kv_tier_capacity_pages"] == 8
    assert backend._config["dram_kv_tier_profile"] == str(profile_path)
    assert backend._dram_kv_tier_profile_bytes == profile_bytes
    assert backend._dram_kv_tier_expected_profile_sha256 == hashlib.sha256(
        profile_bytes
    ).hexdigest()
    assert backend.dram_kv_tier_enabled


def test_process_backend_refuses_respawn_after_profile_mutation(tmp_path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_bytes(b'{"measured":"profile-v1"}')
    backend = ZmqEngineBackend(
        model_path="/model",
        dram_kv_tier_capacity_pages=8,
        dram_kv_tier_profile=profile_path,
    )

    profile_path.write_bytes(b'{"measured":"profile-v2"}')

    with pytest.raises(EngineServiceError, match="changed after backend construction"):
        backend._validate_dram_kv_profile_unchanged()


def test_child_materializes_and_removes_the_pinned_profile_snapshot(tmp_path) -> None:
    original_path = tmp_path / "operator-profile.json"
    pinned_bytes = b'{"measured":"pinned"}'
    config = {"dram_kv_tier_profile": str(original_path), "num_pages": 8}

    with _engine_config_with_pinned_dram_profile(
        config,
        pinned_bytes,
    ) as child_config:
        snapshot_path = Path(child_config["dram_kv_tier_profile"])
        assert child_config is not config
        assert snapshot_path != original_path
        assert snapshot_path.read_bytes() == pinned_bytes
        assert config["dram_kv_tier_profile"] == str(original_path)

    assert not snapshot_path.exists()


def test_process_startup_validates_the_child_dram_tier_identity() -> None:
    actual = _dram_kv_tier_from_startup_frame(
        {
            "dram_kv_tier": {
                "enabled": True,
                "capacity_pages": 8,
                "profile_sha256": "a" * 64,
                "min_restore_tokens": 128,
            }
        }
    )
    assert actual is not None
    _validate_startup_dram_kv_tier_identity(8, actual, "a" * 64)

    with pytest.raises(EngineServiceError, match="request mismatch"):
        _validate_startup_dram_kv_tier_identity(4, actual)
    with pytest.raises(EngineServiceError, match="did not report"):
        _validate_startup_dram_kv_tier_identity(8, None)
    with pytest.raises(EngineServiceError, match="profile mismatch"):
        _validate_startup_dram_kv_tier_identity(8, actual, "b" * 64)
