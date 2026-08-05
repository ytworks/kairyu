"""Focused checkout-wrapper tests for the B7 native KV companion gate."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from bench import kv_answer_equivalence_bench as wrapper
from kairyu.bench import kv_equivalence as gate

ROOT = Path(__file__).resolve().parents[2]


class _FakeTokenizer:
    def __init__(self, size: int = 256) -> None:
        self._decoded = tuple(f" {chr(0x4E00 + index)}" for index in range(size))
        self._encoded = {piece: index for index, piece in enumerate(self._decoded)}
        self._vocab = tuple(f"raw-piece-{index}" for index in range(size))

    def vocab(self) -> list[str]:
        return list(self._vocab)

    def decode(self, token_ids) -> str:
        return "".join(self._decoded[int(token_id)] for token_id in token_ids)

    def encode(self, text: str) -> tuple[int, ...]:
        result: list[int] = []
        position = 0
        while position < len(text):
            piece = text[position : position + 2]
            if piece not in self._encoded:
                raise ValueError(f"unknown fake token piece {piece!r}")
            result.append(self._encoded[piece])
            position += 2
        return tuple(result)


class _FakeCache:
    def __init__(self, backend: _FakeBackend) -> None:
        self.backend = backend
        self.dram_tier_stats = {field: 0 for field in gate.TIER_STATS_FIELDS}

    def peek_cached_tokens(self, token_ids: tuple[int, ...]) -> int:
        if self.backend.main_prompt == token_ids and self.backend.main_resident:
            return len(token_ids)
        return 0


class _FakeBackend:
    def __init__(self, *, cell_id: str, mismatch_warm: bool = False) -> None:
        spec = gate.CELL_SPECS[cell_id]
        self.tiered = bool(spec["dram_required"])
        self.mismatch_warm = mismatch_warm
        self.main_prompt: tuple[int, ...] | None = None
        self.main_resident = False
        self.dram_kv_tier_min_restore_tokens = int(spec["dram_min_restore_tokens"])
        self._cache = _FakeCache(self)

    async def generate(self, request: SimpleNamespace) -> SimpleNamespace:
        prompt = tuple(request.prompt_token_ids)
        phase = request.phase
        if phase == "cold":
            self.main_prompt = prompt
            self.main_resident = True
            cached_tokens = 0
        elif phase == "pressure":
            assert self.main_prompt is not None
            self.main_resident = False
            self._cache.dram_tier_stats["offload_pages"] += len(self.main_prompt) // 16
            cached_tokens = 0
        else:
            assert phase == "warm" and self.main_prompt == prompt
            if self.tiered and not self.main_resident:
                self._cache.dram_tier_stats["restore_attempts"] += 1
                self._cache.dram_tier_stats["restore_pages"] += len(prompt) // 16
            self.main_resident = True
            cached_tokens = len(prompt)

        output = list(range(gate.OUTPUT_TOKENS))
        if phase == "warm" and self.mismatch_warm:
            output[-1] += 1
        return SimpleNamespace(
            finished=True,
            prompt_token_ids=prompt,
            completions=(
                SimpleNamespace(
                    token_ids=tuple(output),
                    text="|".join(str(token_id) for token_id in output),
                    finish_reason="length",
                ),
            ),
            usage=SimpleNamespace(
                prompt_tokens=len(prompt),
                completion_tokens=gate.OUTPUT_TOKENS,
                cached_tokens=cached_tokens,
            ),
        )


def _request_factory(request_id: str, token_ids: tuple[int, ...]) -> SimpleNamespace:
    phase = next(phase for phase in ("cold", "pressure", "warm") if f"-{phase}-" in request_id)
    return SimpleNamespace(
        request_id=request_id,
        phase=phase,
        prompt_token_ids=token_ids,
    )


def _checkpoint() -> dict[str, object]:
    return {
        "model": gate.MODEL,
        "revision": gate.MODEL_REVISION,
        "architecture": dict(gate.EXPECTED_ARCHITECTURE),
        "weights": [dict(row) for row in gate.EXPECTED_WEIGHTS],
        "weights_sha256": gate.EXPECTED_WEIGHTS_SHA256,
        "pinned_weights_rollup": gate.EXPECTED_WEIGHTS_ROLLUP,
        "metadata_sha256": dict(gate.EXPECTED_METADATA_SHA256),
        "weight_index_sha256": gate.EXPECTED_WEIGHT_INDEX_SHA256,
        "model_config_sha256": gate.EXPECTED_MODEL_CONFIG_SHA256,
    }


def _source() -> dict[str, object]:
    files = {path: "a" * 64 for path in gate.REQUIRED_SOURCE_PATHS}
    return {
        "git_head": "b" * 40,
        "tracked_tree_clean": True,
        "file_sha256": files,
        "files_sha256": gate.sha256_json(files),
    }


def _runtime(cell_id: str) -> dict[str, object]:
    spec = gate.CELL_SPECS[cell_id]
    engine_fields = (
        "tensor_parallel_size",
        "page_size",
        "num_pages",
        "dram_profile_sha256",
        "dram_capacity_pages",
        "dram_min_restore_tokens",
        "decode_mode",
        "max_num_batched_tokens",
        "max_model_len",
    )
    return {
        "run_nonce": gate.sha256_json(["nonce", cell_id])[:32],
        "source": _source(),
        "checkpoint": _checkpoint(),
        "engine": {
            "cell_id": cell_id,
            **{field: spec[field] for field in engine_fields},
        },
    }


def _parent(cell_id: str) -> dict[str, object]:
    spec = gate.CELL_SPECS[cell_id]
    kind = spec["parent_binding_kind"]
    if kind == "direct-partial":
        binding: dict[str, object] = {
            "binding_kind": kind,
            "model": gate.MODEL,
            "revision": gate.MODEL_REVISION,
            "pinned_weights_rollup": gate.EXPECTED_WEIGHTS_ROLLUP,
        }
    elif kind == "aggregate-common-f2c":
        binding = {"binding_kind": kind}
    else:
        checkpoint = _checkpoint()
        checkpoint.pop("weight_index_sha256")
        binding = {"binding_kind": kind, **checkpoint}
    feature = str(spec["feature_id"])
    manifest_identity = feature if feature != "f4a" else "f4a-common"
    return {
        "gate": spec["parent_gate"],
        "schema_version": spec["parent_schema_version"],
        "manifest_sha256": gate.sha256_json(["manifest", manifest_identity]),
        "raw_sha256": gate.sha256_json(["raw", cell_id]),
        "quality_manifest_sha256": (
            gate.sha256_json(["quality-manifest", cell_id]) if cell_id == "f4b-tp4" else None
        ),
        "quality_raw_sha256": (
            gate.sha256_json(["quality-raw", cell_id]) if cell_id == "f4b-tp4" else None
        ),
        "parent_topology": dict(spec["parent_topology"]),
        "checkpoint_binding": binding,
    }


async def _capture(cell_id: str, *, mismatch_warm: bool = False):
    spec = gate.CELL_SPECS[cell_id]
    return await wrapper._capture_backend(
        cell_id=cell_id,
        backend=_FakeBackend(cell_id=cell_id, mismatch_warm=mismatch_warm),
        tokenizer=_FakeTokenizer(),
        request_factory=_request_factory,
        parent_evidence=_parent(cell_id),
        runtime=_runtime(cell_id),
        sampling=gate.frozen_config()["sampling"],
        page_size=int(spec["page_size"]),
        num_pages=int(spec["num_pages"]),
    )


@pytest.mark.parametrize("cell_id", gate.CELL_IDS)
def test_prompt_count_is_exact_fixed_1024_for_every_cell(cell_id: str) -> None:
    spec = gate.CELL_SPECS[cell_id]

    assert (
        wrapper._prompt_token_count(
            feature=str(spec["feature_id"]),
            page_size=int(spec["page_size"]),
            num_pages=int(spec["num_pages"]),
            minimum_restore_tokens=int(spec["dram_min_restore_tokens"]),
        )
        == gate.COMPARISON_PROMPT_TOKENS
        == 1024
    )


def test_prompt_count_rejects_insufficient_capacity_or_excess_restore_threshold() -> None:
    with pytest.raises(ValueError, match="cannot fit the fixed comparison prompt"):
        wrapper._prompt_token_count(
            feature="f2c",
            page_size=16,
            num_pages=66,
            minimum_restore_tokens=0,
        )
    with pytest.raises(ValueError, match="threshold exceeds"):
        wrapper._prompt_token_count(
            feature="f4a",
            page_size=16,
            num_pages=8192,
            minimum_restore_tokens=1040,
        )


def test_radix_capture_retains_runtime_nonce_exact_ids_and_full_hit() -> None:
    rows = asyncio.run(_capture("f2c-tp2"))
    requests = [row for row in rows if row["row_type"] == "request"]

    assert [row["phase"] for row in requests] == ["cold", "warm"]
    assert rows[0]["cell_id"] == "f2c-tp2"
    assert rows[0]["prompt"]["token_count"] % 16 == 0
    assert rows[0]["prompt"]["token_ids"] == requests[0]["prompt"]["token_ids"]
    assert all(row["runtime_nonce"] == rows[0]["runtime"]["run_nonce"] for row in requests)
    assert requests[0]["response"]["token_pieces"] == [
        f"raw-piece-{index}" for index in range(gate.OUTPUT_TOKENS)
    ]
    assert requests[0]["response"]["usage"]["cached_tokens"] == 0
    assert requests[1]["response"]["usage"]["cached_tokens"] == rows[0]["prompt"]["token_count"]
    assert wrapper._capture_passed(rows)


def test_radix_proof_uses_native_cache_residency_not_reported_usage() -> None:
    cell_id = "f2c-tp2"
    spec = gate.CELL_SPECS[cell_id]
    backend = _FakeBackend(cell_id=cell_id)
    backend._cache.peek_cached_tokens = lambda _token_ids: 0

    rows = asyncio.run(
        wrapper._capture_backend(
            cell_id=cell_id,
            backend=backend,
            tokenizer=_FakeTokenizer(),
            request_factory=_request_factory,
            parent_evidence=_parent(cell_id),
            runtime=_runtime(cell_id),
            sampling=gate.frozen_config()["sampling"],
            page_size=int(spec["page_size"]),
            num_pages=int(spec["num_pages"]),
        )
    )
    warm = next(row for row in rows if row.get("phase") == "warm")

    assert warm["response"]["usage"]["cached_tokens"] == rows[0]["prompt"]["token_count"]
    assert rows[-1]["feature_proof"]["cache_hit_tokens"] == 0
    assert wrapper._capture_passed(rows) is False


@pytest.mark.parametrize("cell_id", ["f4a-tp4", "f4a-tp8", "f4b-tp4"])
def test_dram_capture_attributes_pressure_and_restore_with_1024_token_floor(
    cell_id: str,
) -> None:
    rows = asyncio.run(_capture(cell_id))
    requests = [row for row in rows if row["row_type"] == "request"]
    proof = rows[-1]["feature_proof"]

    assert [row["phase"] for row in requests] == ["cold", "pressure", "warm"]
    assert rows[0]["prompt"]["token_count"] >= gate.MIN_DRAM_PROMPT_TOKENS
    assert requests[1]["prompt"]["token_ids"] != rows[0]["prompt"]["token_ids"]
    assert proof["target_cached_tokens_pre_warm"] == 0
    assert proof["pressure_delta"]["offload_pages"] > 0
    assert proof["pressure_delta"]["restore_pages"] == 0
    assert proof["warm_delta"]["restore_pages"] > 0
    assert proof["warm_delta"]["restore_attempts"] > 0
    assert wrapper._capture_passed(rows)


def test_capture_equality_failure_is_retained_instead_of_relabelled() -> None:
    rows = asyncio.run(_capture("f2d-tp2", mismatch_warm=True))

    assert rows[-2]["response"]["status"] == "success"
    assert not wrapper._capture_passed(rows)


def test_native_fractional_output_ids_are_rejected_instead_of_truncated() -> None:
    class FractionalBackend(_FakeBackend):
        async def generate(self, request: SimpleNamespace) -> SimpleNamespace:
            result = await super().generate(request)
            completion = result.completions[0]
            completion.token_ids = tuple(token_id + 0.75 for token_id in completion.token_ids)
            return result

    observation = asyncio.run(
        wrapper._request_observation(
            backend=FractionalBackend(cell_id="f2c-tp2"),
            tokenizer=_FakeTokenizer(),
            request_factory=_request_factory,
            cell_id="f2c-tp2",
            phase="cold",
            ordinal=0,
            prompt_token_ids=(1,) * gate.COMPARISON_PROMPT_TOKENS,
            runtime_nonce="a" * 32,
        )
    )

    assert observation["response"]["status"] == "error"
    assert observation["error_type"] == "ValueError"
    assert observation["response"]["token_ids"] == []


@pytest.mark.parametrize("value", [0.0, False, -1, 17])
def test_native_cache_probe_rejects_coerced_or_out_of_range_counts(value: object) -> None:
    prompt = (1,) * 16

    with pytest.raises(ValueError, match="cache-residency probe"):
        wrapper._cached_prefix_probe(lambda _tokens: value, prompt)


def test_five_topology_captures_assemble_verify_and_raw_replay(tmp_path: Path) -> None:
    paths: dict[str, Path] = {}
    for cell_id in gate.CELL_IDS:
        path = tmp_path / f"{cell_id}.jsonl"
        gate.write_jsonl(path, asyncio.run(_capture(cell_id)))
        paths[cell_id] = path

    rows = wrapper._assemble_rows(paths)
    artifact = tmp_path / "artifact"
    raw = artifact / gate.RAW_NAME
    manifest = artifact / gate.MANIFEST_NAME
    written = gate.write_artifact(raw, manifest, rows, assert_gate=True)

    assert written["passed"] is True
    assert [cell["cell_id"] for cell in written["cells"]] == list(gate.CELL_IDS)
    assert gate.verify_artifact(raw, manifest, assert_gate=True) == written
    assert gate.replay_raw(raw, assert_gate=True) == written


def test_assemble_retains_semantic_fail_as_failed_manifest(tmp_path: Path) -> None:
    paths: dict[str, Path] = {}
    for cell_id in gate.CELL_IDS:
        path = tmp_path / f"{cell_id}.jsonl"
        gate.write_jsonl(
            path,
            asyncio.run(_capture(cell_id, mismatch_warm=cell_id == "f2c-tp2")),
        )
        paths[cell_id] = path

    rows = wrapper._assemble_rows(paths)
    manifest = gate.recompute_manifest(rows, raw_sha256="c" * 64)

    assert manifest["passed"] is False
    assert manifest["cells"][0]["passed"] is False


def _write_f2d_parent(tmp_path: Path) -> tuple[Path, Path]:
    mirrored = {
        "schema_version": gate.CELL_SPECS["f2d-tp2"]["parent_schema_version"],
        "gate": "G5-F2d",
        "config": {"profile": "formal", "replicas": 1},
        "inputs": {},
        "trace": {},
        "source": {},
        "source_end": {},
        "environment": {"cpu_count": 1},
    }
    raw = tmp_path / "prefix-weight-f2d-raw.jsonl"
    gate.write_jsonl(
        raw,
        [
            {
                "type": "run",
                **mirrored,
                "row_index": 0,
            }
        ],
    )
    router = tmp_path / "prefix-weight-f2d-router.jsonl"
    gate.write_jsonl(
        router,
        [
            {
                "kind": "replica",
                "session_sha256": "a" * 64,
                "replica": 0,
                "reason": "prefix_match",
                "replica_id": "replica-000",
                "replica_generation": "generation",
                "placement_latency_ns": 1,
                "request_id": "request",
                "placement_started_ns": 1,
                "selected_at_ns": 2,
                "pool_size": 1,
                "eligible_size": 1,
            }
        ],
    )
    manifest = tmp_path / "prefix-weight-f2d-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                **mirrored,
                "passed": True,
                "raw": {"sha256": gate.sha256_file(raw)},
                "router": {"sha256": gate.sha256_file(router)},
                "metrics": {},
            }
        ),
        encoding="utf-8",
    )
    return manifest, raw


def test_parent_context_runs_owner_verifier_and_binds_canonical_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, raw = _write_f2d_parent(tmp_path)
    calls: list[str] = []

    def verify(**kwargs) -> None:
        calls.append(str(kwargs["feature"]))

    monkeypatch.setattr(wrapper, "_verify_parent_artifact", verify)
    evidence, minimum, profile = wrapper._parent_context("f2d-tp2", manifest, raw)

    assert calls == ["f2d"]
    assert evidence["checkpoint_binding"] == {"binding_kind": "aggregate-common-f2c"}
    assert evidence["parent_topology"] == gate.CELL_SPECS["f2d-tp2"]["parent_topology"]
    assert minimum is None and profile is None


def test_f2d_parent_rejects_float_row_count_that_its_owner_coerces_equal(
    tmp_path: Path,
) -> None:
    source = ROOT / "bench/results/f2d-prefix-weight-replay-2026-07-29"
    names = (
        "prefix-weight-f2d-manifest.json",
        "prefix-weight-f2d-raw.jsonl",
        "prefix-weight-f2d-router.jsonl",
    )
    for name in names:
        shutil.copy2(source / name, tmp_path / name)
    manifest_path = tmp_path / names[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["raw"]["rows"] = float(manifest["raw"]["rows"])
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="raw descriptor types differ"):
        wrapper._parent_context(
            "f2d-tp2",
            manifest_path,
            tmp_path / names[1],
        )


def test_f2d_parent_rejects_manifest_raw_environment_type_drift(
    tmp_path: Path,
) -> None:
    source = ROOT / "bench/results/f2d-prefix-weight-replay-2026-07-29"
    names = (
        "prefix-weight-f2d-manifest.json",
        "prefix-weight-f2d-raw.jsonl",
        "prefix-weight-f2d-router.jsonl",
    )
    for name in names:
        shutil.copy2(source / name, tmp_path / name)
    manifest_path = tmp_path / names[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["environment"]["cpu_count"] = float(manifest["environment"]["cpu_count"])
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="run row differs from its manifest"):
        wrapper._parent_context(
            "f2d-tp2",
            manifest_path,
            tmp_path / names[1],
        )


def test_parent_context_rejects_manifest_mutated_by_owner_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, raw = _write_f2d_parent(tmp_path)

    def mutate_manifest(**_kwargs) -> None:
        manifest.write_bytes(manifest.read_bytes() + b" ")

    monkeypatch.setattr(wrapper, "_verify_parent_artifact", mutate_manifest)
    with pytest.raises(ValueError, match="parent manifest changed during validation"):
        wrapper._parent_context("f2d-tp2", manifest, raw)


def test_parent_context_rejects_raw_mutated_by_owner_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, raw = _write_f2d_parent(tmp_path)

    def mutate_raw(**_kwargs) -> None:
        raw.write_bytes(raw.read_bytes() + b"{}\n")

    monkeypatch.setattr(wrapper, "_verify_parent_artifact", mutate_raw)
    with pytest.raises(ValueError, match="parent raw changed during validation"):
        wrapper._parent_context("f2d-tp2", manifest, raw)


def test_parent_context_rejects_raw_mutated_during_binding_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, raw = _write_f2d_parent(tmp_path)
    monkeypatch.setattr(wrapper, "_verify_parent_artifact", lambda **_kwargs: None)

    def mutate_during_binding(**_kwargs):
        raw.write_bytes(raw.read_bytes() + b"{}\n")
        return (
            {"binding_kind": "aggregate-common-f2c"},
            dict(gate.CELL_SPECS["f2d-tp2"]["parent_topology"]),
            None,
            None,
        )

    monkeypatch.setattr(wrapper, "_parent_checkpoint_binding", mutate_during_binding)
    with pytest.raises(ValueError, match="parent raw changed during validation"):
        wrapper._parent_context("f2d-tp2", manifest, raw)


def test_parent_context_rejects_f4a_profile_mutated_by_owner_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell_id = "f4a-tp4"
    spec = gate.CELL_SPECS[cell_id]
    raw = tmp_path / "dram-kv-tier-qwen3-32b-tp4-raw.jsonl"
    gate.write_jsonl(raw, [{"type": "synthetic-parent-row"}])
    manifest = tmp_path / "dram-kv-tier-qwen3-32b-manifest.json"
    manifest.write_text(
        gate.canonical_json(
            {
                "schema_version": spec["parent_schema_version"],
                "gate": spec["parent_gate"],
                "passed": True,
                "raw": {"sha256": gate.sha256_file(raw)},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    profile = tmp_path / "dram-kv-tier-qwen3-32b-tp4-profile.json"
    profile.write_text("{}\n", encoding="utf-8")

    def mutate_profile(**_kwargs) -> None:
        profile.write_text('{"mutated":true}\n', encoding="utf-8")

    monkeypatch.setattr(wrapper, "_verify_parent_artifact", mutate_profile)
    monkeypatch.setattr(
        wrapper,
        "_parent_checkpoint_binding",
        lambda **_kwargs: (
            {"binding_kind": "direct-full"},
            dict(spec["parent_topology"]),
            int(spec["dram_min_restore_tokens"]),
            None,
        ),
    )
    with pytest.raises(ValueError, match="parent DRAM profile changed"):
        wrapper._parent_context(cell_id, manifest, raw)


@pytest.mark.parametrize(
    ("retain_quality_manifest", "message"),
    [
        (False, "parent quality manifest does not exist"),
        (True, "parent quality raw does not exist"),
    ],
)
def test_f4b_parent_requires_both_sibling_quality_files(
    tmp_path: Path,
    retain_quality_manifest: bool,
    message: str,
) -> None:
    manifest = tmp_path / "agentic-kv-tier-f4b-manifest.json"
    raw = tmp_path / "agentic-kv-tier-f4b-raw.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    raw.write_text("{}\n", encoding="utf-8")
    if retain_quality_manifest:
        (tmp_path / wrapper._F4B_QUALITY_MANIFEST_NAME).write_text(
            "{}\n",
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match=message):
        wrapper._parent_context("f4b-tp4", manifest, raw)


def test_f4b_parent_rejects_failed_quality_owner_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supplied = {"passed": True}
    owner = ModuleType("bench.agentic_kv_tier_f4b_bench")
    owner.verify_artifact = lambda *_args, **_kwargs: supplied
    owner.verify_quality_artifact = lambda *_args, **_kwargs: {
        "performance": supplied,
        "quality": {"passed": False},
        "passed": False,
    }
    bench_package = sys.modules["bench"]
    monkeypatch.setitem(sys.modules, owner.__name__, owner)
    monkeypatch.setattr(
        bench_package,
        "agentic_kv_tier_f4b_bench",
        owner,
        raising=False,
    )

    with pytest.raises(ValueError, match=r"performance\+quality parent did not pass"):
        wrapper._verify_parent_artifact(
            feature="f4b",
            manifest_path=tmp_path / "agentic-kv-tier-f4b-manifest.json",
            supplied_manifest=supplied,
            supplied_quality_manifest={"passed": False},
        )


def test_f4b_quality_owner_comparison_rejects_bool_number_coercion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supplied_performance = {"passed": True}
    recomputed_quality = {"passed": True, "request_count": 1}
    supplied_quality = {"passed": 1, "request_count": 1}
    owner = ModuleType("bench.agentic_kv_tier_f4b_bench")
    owner.verify_artifact = lambda *_args, **_kwargs: supplied_performance
    owner.verify_quality_artifact = lambda *_args, **_kwargs: {
        "performance": supplied_performance,
        "quality": recomputed_quality,
        "passed": True,
    }
    bench_package = sys.modules["bench"]
    monkeypatch.setitem(sys.modules, owner.__name__, owner)
    monkeypatch.setattr(
        bench_package,
        "agentic_kv_tier_f4b_bench",
        owner,
        raising=False,
    )

    with pytest.raises(ValueError, match="quality manifest differs from owner replay"):
        wrapper._verify_parent_artifact(
            feature="f4b",
            manifest_path=tmp_path / "agentic-kv-tier-f4b-manifest.json",
            supplied_manifest=supplied_performance,
            supplied_quality_manifest=supplied_quality,
        )


@pytest.mark.parametrize(
    ("filename", "message"),
    [
        (
            wrapper._F4B_QUALITY_MANIFEST_NAME,
            "parent quality manifest changed during validation",
        ),
        (
            wrapper._F4B_QUALITY_RAW_NAME,
            "parent quality raw changed during validation",
        ),
    ],
)
def test_f4b_parent_rejects_quality_swap_during_owner_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    message: str,
) -> None:
    manifest = tmp_path / "agentic-kv-tier-f4b-manifest.json"
    raw = tmp_path / "agentic-kv-tier-f4b-raw.jsonl"
    quality_manifest = tmp_path / wrapper._F4B_QUALITY_MANIFEST_NAME
    quality_raw = tmp_path / wrapper._F4B_QUALITY_RAW_NAME
    manifest.write_text("{}\n", encoding="utf-8")
    raw.write_text("{}\n", encoding="utf-8")
    quality_manifest.write_text('{"passed":true}\n', encoding="utf-8")
    quality_raw.write_text("{}\n", encoding="utf-8")

    def swap_quality(**_kwargs) -> None:
        (tmp_path / filename).write_text('{"swapped":true}\n', encoding="utf-8")

    monkeypatch.setattr(wrapper, "_verify_parent_artifact", swap_quality)
    with pytest.raises(ValueError, match=message):
        wrapper._parent_context("f4b-tp4", manifest, raw)


@pytest.mark.parametrize(
    ("quality_raw_bytes", "message"),
    [
        (b'{"row_index":0,"x":1,"x":1}\n', "duplicate JSON key"),
        (b'{"row_index":0,"x":NaN}\n', "non-finite JSON number"),
        (b'{"row_index": 0}\n', "is not canonical"),
    ],
)
def test_f4b_quality_raw_snapshot_is_strict_canonical_jsonl(
    tmp_path: Path,
    quality_raw_bytes: bytes,
    message: str,
) -> None:
    manifest = tmp_path / "agentic-kv-tier-f4b-manifest.json"
    raw = tmp_path / "agentic-kv-tier-f4b-raw.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    raw.write_text("{}\n", encoding="utf-8")
    (tmp_path / wrapper._F4B_QUALITY_MANIFEST_NAME).write_text(
        "{}\n",
        encoding="utf-8",
    )
    (tmp_path / wrapper._F4B_QUALITY_RAW_NAME).write_bytes(quality_raw_bytes)

    with pytest.raises(ValueError, match=message):
        wrapper._parent_context("f4b-tp4", manifest, raw)


_F2C_DIR = ROOT / "bench/results/f2c-kv-aware-ttft-qwen3-32b-2026-07-29"
_F2D_DIR = ROOT / "bench/results/f2d-prefix-weight-replay-2026-07-29"
_F4A_DIR = ROOT / "bench/results/g5-f4a-dram-kv-tier-qwen3-32b-rtxpro6000-2026-08-01"
_F4B_DIR = ROOT / "bench/results/g5-f4b-agentic-kv-tier-qwen3-32b-rtxpro6000-2026-08-01"


def test_f2d_parent_row_schemas_reject_score_and_router_type_coercion() -> None:
    manifest = wrapper._strict_json_object(_F2D_DIR / "prefix-weight-f2d-manifest.json")
    raw_rows = wrapper._strict_jsonl_bytes(
        (_F2D_DIR / "prefix-weight-f2d-raw.jsonl").read_bytes(),
        description="retained F2d raw",
    )
    router_rows = wrapper._strict_jsonl_bytes(
        (_F2D_DIR / "prefix-weight-f2d-router.jsonl").read_bytes(),
        description="retained F2d router",
        canonical=False,
    )
    wrapper._validate_f2d_parent_rows(manifest, raw_rows, router_rows)

    bool_tick_rows = json.loads(json.dumps(raw_rows))
    outcome = next(row for row in bool_tick_rows if row.get("kind") == "placement_outcome")
    outcome["ttft_ticks"] = True
    with pytest.raises(ValueError, match="outcome row .* field types differ"):
        wrapper._validate_f2d_parent_rows(manifest, bool_tick_rows, router_rows)

    negative_zero_rows = json.loads(json.dumps(raw_rows))
    outcome = next(row for row in negative_zero_rows if row.get("kind") == "placement_outcome")
    outcome["beta"] = -0.0
    with pytest.raises(ValueError, match="outcome row .* field types differ"):
        wrapper._validate_f2d_parent_rows(manifest, negative_zero_rows, router_rows)

    float_router_rows = json.loads(json.dumps(router_rows))
    float_router_rows[0]["pool_size"] = float(float_router_rows[0]["pool_size"])
    with pytest.raises(ValueError, match="router row 0 field types differ"):
        wrapper._validate_f2d_parent_rows(manifest, raw_rows, float_router_rows)

    coordinated_manifest = json.loads(json.dumps(manifest))
    coordinated_rows = json.loads(json.dumps(raw_rows))
    coordinated_manifest["environment"]["cpu_count"] = float(
        coordinated_manifest["environment"]["cpu_count"]
    )
    coordinated_rows[0]["environment"]["cpu_count"] = float(
        coordinated_rows[0]["environment"]["cpu_count"]
    )
    with pytest.raises(ValueError, match="cpu_count must be an exact integer"):
        wrapper._validate_f2d_parent_rows(
            coordinated_manifest,
            coordinated_rows,
            router_rows,
        )


def test_parent_row_type_schemas_reject_bool_int_aliases() -> None:
    f2c_rows = wrapper._strict_jsonl_bytes(
        (_F2C_DIR / "kv-aware-ttft-f2c-raw.jsonl").read_bytes(),
        description="retained F2c raw",
    )
    request = next(row for row in f2c_rows if row.get("type") == "request")
    request["replica_ordinal"] = False
    with pytest.raises(ValueError, match="replica_ordinal must be an exact integer"):
        wrapper._validate_f2c_parent_rows(f2c_rows)

    f4a_rows = wrapper._strict_jsonl_bytes(
        (_F4A_DIR / "dram-kv-tier-qwen3-32b-tp4-raw.jsonl").read_bytes(),
        description="retained F4a raw",
    )
    snapshot = next(row for row in f4a_rows if row.get("row_type") == "snapshot")
    snapshot["ownership"]["before"]["source_device_owned_pages"] = 0.0
    with pytest.raises(ValueError, match="source_device_owned_pages must be an exact integer"):
        wrapper._validate_f4a_parent_rows(f4a_rows)

    quality_rows = wrapper._strict_jsonl_bytes(
        (_F4B_DIR / wrapper._F4B_QUALITY_RAW_NAME).read_bytes(),
        description="retained F4b quality raw",
    )
    quality_rows[1]["row_index"] = True
    with pytest.raises(ValueError, match="row indices must be exact contiguous integers"):
        wrapper._validate_f4b_parent_rows(
            quality_rows,
            description="retained F4b quality raw",
        )


def test_f4b_parent_tp_rejects_int_float_type_coercion() -> None:
    rows = wrapper._strict_jsonl_bytes(
        (_F4B_DIR / "agentic-kv-tier-f4b-raw.jsonl").read_bytes(),
        description="retained F4b parent raw",
    )
    first = next(row for row in rows if row.get("row_type") == "shard_start")
    first["config"]["tensor_parallel_size"] = 4.0

    with pytest.raises(ValueError, match="TP size differs"):
        wrapper._parent_checkpoint_binding(
            cell_id="f4b-tp4",
            manifest={},
            rows=rows,
        )


@pytest.mark.parametrize(
    ("cell_id", "manifest", "raw", "profile"),
    [
        (
            "f2c-tp2",
            _F2C_DIR / "kv-aware-ttft-f2c-manifest.json",
            _F2C_DIR / "kv-aware-ttft-f2c-raw.jsonl",
            None,
        ),
        (
            "f2d-tp2",
            _F2D_DIR / "prefix-weight-f2d-manifest.json",
            _F2D_DIR / "prefix-weight-f2d-raw.jsonl",
            None,
        ),
        (
            "f4a-tp4",
            _F4A_DIR / "dram-kv-tier-qwen3-32b-manifest.json",
            _F4A_DIR / "dram-kv-tier-qwen3-32b-tp4-raw.jsonl",
            _F4A_DIR / "dram-kv-tier-qwen3-32b-tp4-profile.json",
        ),
        (
            "f4a-tp8",
            _F4A_DIR / "dram-kv-tier-qwen3-32b-manifest.json",
            _F4A_DIR / "dram-kv-tier-qwen3-32b-tp8-raw.jsonl",
            _F4A_DIR / "dram-kv-tier-qwen3-32b-tp8-profile.json",
        ),
        (
            "f4b-tp4",
            _F4B_DIR / "agentic-kv-tier-f4b-manifest.json",
            _F4B_DIR / "agentic-kv-tier-f4b-raw.jsonl",
            _F4A_DIR / "dram-kv-tier-qwen3-32b-tp4-profile.json",
        ),
    ],
)
def test_retained_parent_owner_replay_checkpoint_and_profile_binding(
    cell_id: str,
    manifest: Path,
    raw: Path,
    profile: Path | None,
) -> None:
    evidence, minimum, profile_descriptor = wrapper._parent_context(cell_id, manifest, raw)
    spec = gate.CELL_SPECS[cell_id]

    assert evidence["checkpoint_binding"]["binding_kind"] == spec["parent_binding_kind"]
    assert evidence["parent_topology"] == spec["parent_topology"]
    if cell_id == "f4b-tp4":
        assert evidence["quality_manifest_sha256"] == gate.sha256_file(
            manifest.parent / wrapper._F4B_QUALITY_MANIFEST_NAME
        )
        assert evidence["quality_raw_sha256"] == gate.sha256_file(
            manifest.parent / wrapper._F4B_QUALITY_RAW_NAME
        )
    else:
        assert evidence["quality_manifest_sha256"] is None
        assert evidence["quality_raw_sha256"] is None
    if profile is None:
        assert minimum is None and profile_descriptor is None
    else:
        validated = wrapper._validate_dram_profile(
            cell_id=cell_id,
            profile_path=profile,
            checkpoint=_checkpoint(),
            parent_evidence=evidence,
            expected_min_restore_tokens=minimum,
            expected_parent_profile=profile_descriptor,
        )
        assert validated["policy"]["min_restore_tokens"] == spec["dram_min_restore_tokens"]
        assert gate.sha256_file(profile) == spec["dram_profile_sha256"]


def test_dram_profile_rejects_mutation_during_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell_id = "f4a-tp4"
    spec = gate.CELL_SPECS[cell_id]
    parent = _parent(cell_id)
    profile_value = {
        "schema_version": "kairyu.dram-kv-crossover.v2",
        "passed": True,
        "runtime_identity": {
            "tensor_parallel_size": spec["tensor_parallel_size"],
            "model_config_sha256": gate.EXPECTED_MODEL_CONFIG_SHA256,
        },
        "policy": {"min_restore_tokens": spec["dram_min_restore_tokens"]},
        "evidence_sha256": gate.sha256_json(
            {
                "manifest_sha256": parent["manifest_sha256"],
                "raw_sha256": parent["raw_sha256"],
            }
        ),
    }
    profile = tmp_path / "profile.json"
    profile.write_text(gate.canonical_json(profile_value) + "\n", encoding="utf-8")
    descriptor = {
        "file_sha256": gate.sha256_file(profile),
        "value_sha256": gate.sha256_json(profile_value),
        "value": profile_value,
    }
    original_sha256_json = gate.sha256_json
    mutated = False

    def mutate_during_value_hash(value) -> str:
        nonlocal mutated
        digest = original_sha256_json(value)
        if (
            not mutated
            and isinstance(value, dict)
            and value.get("schema_version") == "kairyu.dram-kv-crossover.v2"
        ):
            mutated = True
            profile.write_text("{}\n", encoding="utf-8")
        return digest

    monkeypatch.setattr(gate, "sha256_json", mutate_during_value_hash)
    with pytest.raises(ValueError, match="DRAM profile changed during validation"):
        wrapper._validate_dram_profile(
            cell_id=cell_id,
            profile_path=profile,
            checkpoint=_checkpoint(),
            parent_evidence=parent,
            expected_min_restore_tokens=int(spec["dram_min_restore_tokens"]),
            expected_parent_profile=descriptor,
        )


def test_parent_owner_replay_failure_precedes_raw_identity_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": gate.CELL_SPECS["f2d-tp2"]["parent_schema_version"],
                "gate": "G5-F2d",
                "passed": True,
            }
        ),
        encoding="utf-8",
    )
    raw = tmp_path / "invalid.jsonl"
    raw.write_text("not-json\n", encoding="utf-8")
    (tmp_path / "prefix-weight-f2d-router.jsonl").write_text("{}\n", encoding="utf-8")

    def reject(**_kwargs) -> None:
        raise ValueError("owner replay rejected")

    monkeypatch.setattr(wrapper, "_verify_parent_artifact", reject)
    with pytest.raises(ValueError, match="owner replay rejected"):
        wrapper._parent_context("f2d-tp2", manifest, raw)


def _f2c_parent_row() -> dict[str, object]:
    services = [
        {"service": f"replica-{cohort}{replica}", "device_ids": [str(gpu), str(gpu + 1)]}
        for cohort, replica, gpu in (("a", 0, 0), ("a", 1, 2), ("b", 0, 4), ("b", 1, 6))
    ]
    return {
        "type": "run",
        "schema_version": gate.CELL_SPECS["f2c-tp2"]["parent_schema_version"],
        "gate": "G5-F2c",
        "model": gate.MODEL,
        "model_revision": gate.MODEL_REVISION,
        "model_digests": {"weights-rollup": gate.EXPECTED_WEIGHTS_ROLLUP},
        "provenance_start": {
            "configuration": {"value": {"runtime_attestation": {"services": services}}}
        },
    }


def test_f2c_parent_raw_proves_four_disjoint_tp2_services() -> None:
    binding, topology, minimum, profile = wrapper._parent_checkpoint_binding(
        cell_id="f2c-tp2", manifest={}, rows=[_f2c_parent_row()]
    )

    assert binding["binding_kind"] == "direct-partial"
    assert topology == gate.CELL_SPECS["f2c-tp2"]["parent_topology"]
    assert minimum is None and profile is None

    row = _f2c_parent_row()
    services = row["provenance_start"]["configuration"]["value"]["runtime_attestation"]["services"]
    services[-1]["device_ids"] = ["0", "7"]
    with pytest.raises(ValueError, match="globally unique"):
        wrapper._parent_checkpoint_binding(cell_id="f2c-tp2", manifest={}, rows=[row])


def _write_fake_checkpoint(path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    path.mkdir()
    architecture = {
        "model_type": "qwen3-test",
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "vocab_size": 64,
    }
    (path / "config.json").write_text(json.dumps(architecture), encoding="utf-8")
    for name in (
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ):
        (path / name).write_text("{}", encoding="utf-8")
    (path / "merges.txt").write_text("merge\n", encoding="utf-8")
    names = [f"model-{index:05d}-of-00017.safetensors" for index in range(1, 18)]
    for index, name in enumerate(names, 1):
        (path / name).write_bytes(f"shard-{index}".encode())
    index_value = {"weight_map": {f"weight-{index}": name for index, name in enumerate(names)}}
    (path / "model.safetensors.index.json").write_text(json.dumps(index_value), encoding="utf-8")
    weights = [
        {
            "file": name,
            "bytes": (path / name).stat().st_size,
            "sha256": gate.sha256_file(path / name),
        }
        for name in names
    ]
    metadata = {name: gate.sha256_file(path / name) for name in wrapper._CHECKPOINT_METADATA}
    expected = {
        "architecture": architecture,
        "weights": weights,
        "weights_sha256": gate.sha256_json(weights),
        "pinned_weights_rollup": gate.legacy_weights_rollup(weights),
        "metadata_sha256": metadata,
        "weight_index_sha256": gate.sha256_file(path / "model.safetensors.index.json"),
        "model_config_sha256": gate.sha256_json(architecture),
    }
    monkeypatch.setattr(gate, "EXPECTED_ARCHITECTURE", architecture)
    monkeypatch.setattr(gate, "EXPECTED_WEIGHTS", tuple(weights))
    monkeypatch.setattr(gate, "EXPECTED_WEIGHTS_SHA256", expected["weights_sha256"])
    monkeypatch.setattr(gate, "EXPECTED_WEIGHTS_ROLLUP", expected["pinned_weights_rollup"])
    monkeypatch.setattr(gate, "EXPECTED_METADATA_SHA256", metadata)
    monkeypatch.setattr(gate, "EXPECTED_WEIGHT_INDEX_SHA256", expected["weight_index_sha256"])
    monkeypatch.setattr(gate, "EXPECTED_MODEL_CONFIG_SHA256", expected["model_config_sha256"])
    return expected


def test_live_checkpoint_hashes_every_shard_and_rejects_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "model"
    expected = _write_fake_checkpoint(model_path, monkeypatch)

    checkpoint = wrapper._live_checkpoint(model_path)
    for key, value in expected.items():
        assert checkpoint[key] == value

    (model_path / "model-00017-of-00017.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="weight inventory differs"):
        wrapper._live_checkpoint(model_path)


def _make_source_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "one.py").write_text("one\n", encoding="utf-8")
    (repo / "two.py").write_text("two\n", encoding="utf-8")
    (repo / ".gitignore").write_text(
        "*.so\n.venv/\n__pycache__/\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "add", ".gitignore", "one.py", "two.py"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=B7 Test",
            "-c",
            "user.email=b7@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=repo,
        check=True,
    )
    monkeypatch.setattr(wrapper, "_REPO_ROOT", repo)
    monkeypatch.setattr(wrapper, "_RUNTIME_SOURCE_PATHS", ("one.py", "two.py"))
    # These fixture repos intentionally contain only the two synthetic source
    # files, not the real checkout-owned modules already loaded by pytest.
    monkeypatch.setattr(wrapper, "_CHECKOUT_MODULE_PREFIXES", ())
    return repo


def test_source_snapshot_requires_tracked_head_identical_required_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _make_source_repo(tmp_path, monkeypatch)

    snapshot = wrapper._source_snapshot()
    assert snapshot["tracked_tree_clean"] is True
    assert list(snapshot["file_sha256"]) == ["one.py", "two.py"]

    (repo / "one.py").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean tracked source"):
        wrapper._source_snapshot()

    subprocess.run(["git", "checkout", "--", "one.py"], cwd=repo, check=True)
    (repo / "untracked.py").write_text("new\n", encoding="utf-8")
    monkeypatch.setattr(wrapper, "_RUNTIME_SOURCE_PATHS", ("one.py", "two.py", "untracked.py"))
    with pytest.raises(ValueError, match="untracked"):
        wrapper._source_snapshot()


def test_preimport_source_check_rejects_tracked_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _make_source_repo(tmp_path, monkeypatch)
    wrapper._assert_tracked_tree_clean_before_repo_import()

    (repo / "one.py").write_text("dirty before import\n", encoding="utf-8")
    with pytest.raises(ValueError, match="before repository import"):
        wrapper._assert_tracked_tree_clean_before_repo_import()


def test_source_snapshot_rejects_repo_root_sitecustomize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _make_source_repo(tmp_path, monkeypatch)
    (repo / "sitecustomize.py").write_text("raise SystemExit\n", encoding="utf-8")

    with pytest.raises(ValueError, match="import artifacts.*sitecustomize.py"):
        wrapper._source_snapshot()


def test_source_snapshot_rejects_untracked_flashinfer_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _make_source_repo(tmp_path, monkeypatch)
    package = repo / "flashinfer"
    package.mkdir()
    (package / "__init__.py").write_text("SHADOW = True\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"flashinfer/__init__\.py"):
        wrapper._source_snapshot()


def test_source_snapshot_rejects_ignored_native_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _make_source_repo(tmp_path, monkeypatch)
    extension = repo / "flashinfer_native.cpython-312-x86_64-linux-gnu.so"
    extension.write_bytes(b"ELF-shadow")

    with pytest.raises(ValueError, match=r"flashinfer_native.*\.so"):
        wrapper._source_snapshot()


@pytest.mark.parametrize("filename", ["direct_shadow.pyc", "runtime_hook.pth"])
def test_source_snapshot_rejects_other_code_bearing_import_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
) -> None:
    repo = _make_source_repo(tmp_path, monkeypatch)
    (repo / filename).write_bytes(b"executable-shadow")

    with pytest.raises(ValueError, match=filename.replace(".", r"\.")):
        wrapper._source_snapshot()


@pytest.mark.parametrize(
    "relative_path",
    ["vendor-hooks/runtime.pth", "bench/results/.snapshot/shadow.py"],
)
def test_source_snapshot_rejects_code_artifacts_below_nonidentifier_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
) -> None:
    repo = _make_source_repo(tmp_path, monkeypatch)
    artifact = repo / relative_path
    artifact.parent.mkdir(parents=True)
    artifact.write_text("shadow\n", encoding="utf-8")

    with pytest.raises(ValueError, match=artifact.name.replace(".", r"\.")):
        wrapper._source_snapshot()


def test_source_snapshot_rejects_untracked_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _make_source_repo(tmp_path, monkeypatch)
    (repo / "runtime-link").symlink_to(repo / "one.py")

    with pytest.raises(ValueError, match="runtime-link"):
        wrapper._source_snapshot()


def test_source_snapshot_allows_only_venv_generated_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _make_source_repo(tmp_path, monkeypatch)
    (repo / ".venv").mkdir()
    (repo / ".venv/sitecustomize.py").write_text("ignored\n", encoding="utf-8")

    assert wrapper._source_snapshot()["tracked_tree_clean"] is True


def test_source_snapshot_rejects_repo_pycache_bytecode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _make_source_repo(tmp_path, monkeypatch)
    cache = repo / "package/__pycache__"
    cache.mkdir(parents=True)
    (cache / "shadow.pyc").write_bytes(b"generated")

    with pytest.raises(ValueError, match=r"__pycache__/shadow\.pyc"):
        wrapper._source_snapshot()


def test_source_snapshot_rejects_loaded_untracked_repo_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _make_source_repo(tmp_path, monkeypatch)
    payload = repo / "payload.data"
    payload.write_bytes(b"loaded executable payload")
    loaded = ModuleType("b7_loaded_payload")
    loaded.__file__ = str(payload)
    monkeypatch.setitem(sys.modules, loaded.__name__, loaded)

    with pytest.raises(ValueError, match="loaded repo module.*untracked"):
        wrapper._source_snapshot()


def test_loaded_checkout_namespace_rejects_an_external_module_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _make_source_repo(tmp_path, monkeypatch)
    monkeypatch.setattr(wrapper, "_CHECKOUT_MODULE_PREFIXES", ("b7_checkout",))
    external = tmp_path / "external/injected.py"
    external.parent.mkdir()
    external.write_text("injected\n", encoding="utf-8")
    loaded = ModuleType("b7_checkout.injected")
    loaded.__file__ = str(external)
    monkeypatch.setitem(sys.modules, loaded.__name__, loaded)

    with pytest.raises(ValueError, match="loaded outside the repository"):
        wrapper._assert_loaded_repo_modules_head_identical()
    assert repo != external.parent


def test_loaded_checkout_namespace_is_not_exempt_inside_repo_venv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = ModuleType("kairyu.venv_injected")
    loaded.__file__ = str(ROOT / ".venv/site-packages/kairyu/venv_injected.py")
    monkeypatch.setitem(sys.modules, loaded.__name__, loaded)
    monkeypatch.setattr(wrapper, "_head_identical_source", lambda *_args, **_kwargs: "")

    with pytest.raises(ValueError, match="outside its tracked package"):
        wrapper._assert_loaded_repo_modules_head_identical()


def _native_args(
    tmp_path: Path,
    *,
    model_path: Path,
) -> SimpleNamespace:
    manifest = tmp_path / "manifest.json"
    raw = tmp_path / "raw.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    raw.write_text("{}\n", encoding="utf-8")
    return SimpleNamespace(
        cell="f2c-tp2",
        model_path=model_path,
        parent_manifest=manifest,
        parent_raw=raw,
        dram_profile=None,
    )


def _install_native_import_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    backend_class: type,
    tokenizer_resolver=None,
) -> None:
    backend_api = ModuleType("kairyu.engine.backend")
    backend_api.GenerationRequest = object
    backend_runtime = ModuleType("kairyu.engine.kairyu_backend")
    backend_runtime.KairyuBackend = backend_class
    prompt_module = ModuleType("kairyu.engine.prompt")
    prompt_module.TokensPrompt = object
    tokenizer_module = ModuleType("kairyu.engine.tokenizer")
    tokenizer_module.resolve_tokenizer = tokenizer_resolver or (lambda _path: object())
    sampling_module = ModuleType("kairyu.sampling_params")
    sampling_module.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
    for name, module in (
        ("kairyu.engine.backend", backend_api),
        ("kairyu.engine.kairyu_backend", backend_runtime),
        ("kairyu.engine.prompt", prompt_module),
        ("kairyu.engine.tokenizer", tokenizer_module),
        ("kairyu.sampling_params", sampling_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)


def test_wrong_checkpoint_stops_run_native_before_source_or_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "wrong-model"
    model_path.mkdir()
    monkeypatch.setattr(
        wrapper,
        "_parent_context",
        lambda *_args: (_parent("f2c-tp2"), None, None),
    )

    def wrong_checkpoint(_path: Path):
        raise ValueError("wrong checkpoint")

    monkeypatch.setattr(wrapper, "_live_checkpoint", wrong_checkpoint)
    monkeypatch.setattr(
        wrapper,
        "_source_snapshot",
        lambda: pytest.fail("source/backend path must not run after checkpoint failure"),
    )

    class FailBackend:
        def __init__(self, **_kwargs) -> None:
            pytest.fail("backend must not be constructed after checkpoint failure")

    backend_module = ModuleType("kairyu.engine.kairyu_backend")
    backend_module.KairyuBackend = FailBackend
    monkeypatch.setitem(sys.modules, "kairyu.engine.kairyu_backend", backend_module)
    args = _native_args(tmp_path, model_path=model_path)

    with pytest.raises(ValueError, match="wrong checkpoint"):
        asyncio.run(wrapper._run_native(args))


def test_backend_validation_failure_still_shuts_down_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    monkeypatch.setattr(
        wrapper,
        "_parent_context",
        lambda *_args: (_parent("f2c-tp2"), None, None),
    )
    monkeypatch.setattr(wrapper, "_source_snapshot", _source)
    monkeypatch.setattr(wrapper, "_live_checkpoint", lambda _path: _checkpoint())

    shutdown = False

    class FakeBackend:
        def __init__(self, **_kwargs) -> None:
            # Deliberately violate the first live-geometry check.
            self.tensor_parallel_size = 999

        async def shutdown(self) -> None:
            nonlocal shutdown
            shutdown = True

    _install_native_import_stubs(
        monkeypatch,
        backend_class=FakeBackend,
    )
    args = _native_args(tmp_path, model_path=model_path)

    with pytest.raises(ValueError, match="TP size differs"):
        asyncio.run(wrapper._run_native(args))
    assert shutdown is True


def test_run_native_rechecks_source_after_shutdown_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _make_source_repo(tmp_path, monkeypatch)
    model_path = tmp_path / "model"
    model_path.mkdir()
    monkeypatch.setattr(
        wrapper,
        "_parent_context",
        lambda *_args: (_parent("f2c-tp2"), None, None),
    )
    monkeypatch.setattr(wrapper, "_live_checkpoint", lambda _path: _checkpoint())
    shutdown = False

    class DriftBackend:
        def __init__(self, **_kwargs) -> None:
            (repo / "one.py").write_text("drift during native run\n", encoding="utf-8")
            self.tensor_parallel_size = 999

        async def shutdown(self) -> None:
            nonlocal shutdown
            shutdown = True

    _install_native_import_stubs(monkeypatch, backend_class=DriftBackend)
    args = _native_args(tmp_path, model_path=model_path)

    with pytest.raises(ValueError, match="clean tracked source"):
        asyncio.run(wrapper._run_native(args))
    assert shutdown is True


def test_run_native_resolves_model_symlink_once_and_rehashes_fixed_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_a = tmp_path / "model-a"
    model_b = tmp_path / "model-b"
    model_a.mkdir()
    model_b.mkdir()
    alias = tmp_path / "model-current"
    alias.symlink_to(model_a, target_is_directory=True)
    monkeypatch.setattr(
        wrapper,
        "_parent_context",
        lambda *_args: (_parent("f2c-tp2"), None, None),
    )
    monkeypatch.setattr(wrapper, "_source_snapshot", _source)
    checkpoint_paths: list[Path] = []

    def checkpoint(path: Path) -> dict[str, object]:
        checkpoint_paths.append(path)
        if len(checkpoint_paths) == 1:
            alias.unlink()
            alias.symlink_to(model_b, target_is_directory=True)
        return _checkpoint()

    monkeypatch.setattr(wrapper, "_live_checkpoint", checkpoint)
    backend_paths: list[Path] = []
    shutdown = False

    class FixedTargetBackend:
        def __init__(self, **kwargs) -> None:
            backend_paths.append(Path(kwargs["model_path"]))
            assert Path(kwargs["tokenizer"]) == model_a.resolve()
            self.tensor_parallel_size = 999

        async def shutdown(self) -> None:
            nonlocal shutdown
            shutdown = True

    tokenizer_paths: list[Path] = []

    def resolve_tokenizer(path: str) -> object:
        tokenizer_paths.append(Path(path))
        return object()

    _install_native_import_stubs(
        monkeypatch,
        backend_class=FixedTargetBackend,
        tokenizer_resolver=resolve_tokenizer,
    )
    args = _native_args(tmp_path, model_path=alias)

    with pytest.raises(ValueError, match="TP size differs"):
        asyncio.run(wrapper._run_native(args))
    assert shutdown is True
    assert checkpoint_paths == [model_a.resolve(), model_a.resolve()]
    assert tokenizer_paths == [model_a.resolve()]
    assert backend_paths == [model_a.resolve()]
    assert alias.resolve() == model_b.resolve()


def test_run_native_rejects_checkpoint_drift_after_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    monkeypatch.setattr(
        wrapper,
        "_parent_context",
        lambda *_args: (_parent("f2c-tp2"), None, None),
    )
    monkeypatch.setattr(wrapper, "_source_snapshot", _source)
    checkpoint_calls = 0

    def drifting_checkpoint(_path: Path) -> dict[str, object]:
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        value = _checkpoint()
        if checkpoint_calls > 1:
            value["revision"] = "mutated-after-preflight"
        return value

    monkeypatch.setattr(wrapper, "_live_checkpoint", drifting_checkpoint)

    class DriftBackend:
        def __init__(self, **_kwargs) -> None:
            self.tensor_parallel_size = 999

        async def shutdown(self) -> None:
            return None

    _install_native_import_stubs(monkeypatch, backend_class=DriftBackend)
    args = _native_args(tmp_path, model_path=model_path)

    with pytest.raises(ValueError, match="checkpoint identity changed"):
        asyncio.run(wrapper._run_native(args))
    assert checkpoint_calls == 2


def test_help_path_does_not_import_native_engine_modules() -> None:
    program = f"""
import runpy
import sys
repo = {str(ROOT)!r}
sys.path[:] = [entry for entry in sys.path if entry != repo]
sys.path.insert(0, repo)
sys.path.insert(0, {str(ROOT / "external-shadow-path")!r})
sys.argv = [{str(ROOT / "bench/kv_answer_equivalence_bench.py")!r}, '--help']
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
except SystemExit as error:
    assert error.code == 0
assert sys.path[0] == repo
for forbidden in (
    'kairyu.engine.kairyu_backend',
    'kairyu.engine.core.model_runner',
    'torch',
):
    assert forbidden not in sys.modules, forbidden
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "run-native" in completed.stdout


def test_direct_cli_reexecs_isolated_before_pythonpath_stdlib_shadow(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "shadow-executed"
    (tmp_path / "argparse.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        "Path(os.environ['KAIRYU_B7_SHADOW_MARKER']).write_text('executed')\n"
        "raise RuntimeError('shadow argparse executed')\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(tmp_path)
    environment["KAIRYU_B7_SHADOW_MARKER"] = str(marker)

    completed = subprocess.run(
        [sys.executable, str(ROOT / "bench/kv_answer_equivalence_bench.py"), "--help"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "run-native" in completed.stdout
    assert not marker.exists()


def test_evidence_cli_requires_initial_isolated_no_bytecode_interpreter(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "shadow-executed"
    (tmp_path / "argparse.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench/kv_answer_equivalence_bench.py"),
            "verify",
            "--artifact",
            str(tmp_path / "missing"),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "must start as: python -I -B" in completed.stderr
    assert not marker.exists()


def test_module_form_is_not_an_evidence_invocation() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "bench.kv_answer_equivalence_bench",
            "verify",
            "--artifact",
            "unused",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "require the direct checkout path" in completed.stderr


def test_wrapper_import_rejects_a_preloaded_noncheckout_gate_module() -> None:
    program = f"""
import runpy
import sys
from kairyu.bench import kv_equivalence
kv_equivalence.__file__ = sys.executable
runpy.run_path(
    {str(ROOT / "bench/kv_answer_equivalence_bench.py")!r},
    run_name='b7_noncheckout_probe',
)
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "refusing non-checkout kairyu.bench.kv_equivalence origin" in completed.stderr


def test_offline_cli_runs_import_artifact_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        wrapper,
        "_assert_no_import_shadows",
        lambda: (_ for _ in ()).throw(ValueError("import artifact preflight")),
    )

    with pytest.raises(ValueError, match="import artifact preflight"):
        wrapper.main(["verify", "--artifact", str(tmp_path / "artifact")])


def test_parser_exposes_exact_five_cell_surface_and_no_free_geometry() -> None:
    parser = wrapper._build_parser()
    argv = ["assemble"]
    for cell_id in gate.CELL_IDS:
        argv.extend((f"--{cell_id}", f"{cell_id}.jsonl"))
    argv.extend(("--output-dir", "artifact"))
    args = parser.parse_args(argv)

    assert args.command == "assemble"
    assert args.output_dir == Path("artifact")
    assert args.f4a_tp8 == Path("f4a-tp8.jsonl")

    native = parser.parse_args(
        [
            "run-native",
            "--cell",
            "f2c-tp2",
            "--model-path",
            "model",
            "--parent-manifest",
            "manifest.json",
            "--parent-raw",
            "raw.jsonl",
            "--output",
            "capture.jsonl",
        ]
    )
    assert native.cell == "f2c-tp2"
    for removed in (
        "feature",
        "tensor_parallel_size",
        "num_pages",
        "page_size",
        "dram_capacity_pages",
    ):
        assert not hasattr(native, removed)


def test_run_native_output_creation_is_exclusive_across_preflight_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "capture.jsonl"
    competitor = b"pre-existing-race-winner\n"

    async def race_after_preflight(_args):
        output.write_bytes(competitor)
        return ([{"row_type": "synthetic"}], True)

    monkeypatch.setattr(wrapper, "_assert_no_import_shadows", lambda: None)
    monkeypatch.setattr(wrapper, "_run_native", race_after_preflight)
    with pytest.raises(ValueError, match="refusing to overwrite existing capture"):
        wrapper.main(
            [
                "run-native",
                "--cell",
                "f2c-tp2",
                "--model-path",
                str(tmp_path / "model"),
                "--parent-manifest",
                str(tmp_path / "manifest.json"),
                "--parent-raw",
                str(tmp_path / "raw.jsonl"),
                "--output",
                str(output),
            ]
        )
    assert output.read_bytes() == competitor


def test_output_directory_reservation_rejects_even_an_existing_empty_directory(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "artifact"
    output_dir.mkdir()

    with pytest.raises(ValueError, match="existing output directory"):
        wrapper._reserve_output_directory(output_dir)
    assert list(output_dir.iterdir()) == []


def test_assemble_validates_inputs_before_reserving_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "artifact"
    monkeypatch.setattr(wrapper, "_assert_no_import_shadows", lambda: None)
    monkeypatch.setattr(
        wrapper,
        "_assemble_rows",
        lambda _paths: (_ for _ in ()).throw(ValueError("invalid child capture")),
    )
    argv = ["assemble"]
    for cell_id in gate.CELL_IDS:
        argv.extend((f"--{cell_id}", str(tmp_path / f"{cell_id}.jsonl")))
    argv.extend(("--output-dir", str(output_dir)))

    with pytest.raises(ValueError, match="invalid child capture"):
        wrapper.main(argv)
    assert not output_dir.exists()
