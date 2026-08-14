"""Focused CPU tests for the replayable G2 A7 real-engine gate."""

from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest

from verification.l1.performance import tp_kv_hit_g2_a7_bench as a7

_SOURCE_COMMIT = "1" * 40
_IMAGE_ID = f"sha256:{'2' * 64}"
_TOKENIZER_SHA256 = a7._expected_checkpoint_identity()["tokenizer_sha256"][
    "tokenizer.json"
]


class _StablePieceTokenizer:
    """Tokenizer fixture whose space-prefixed pieces round-trip one-for-one."""

    _piece = re.compile(r" piece(\d{3})")

    def __init__(self, size: int = 128) -> None:
        self._pieces = [f" piece{token_id:03d}" for token_id in range(size)]

    def vocab(self) -> list[str]:
        return list(self._pieces)

    def decode(self, token_ids) -> str:
        return "".join(self._pieces[token_id] for token_id in token_ids)

    def encode(self, text: str) -> tuple[int, ...]:
        token_ids: list[int] = []
        position = 0
        for match in self._piece.finditer(text):
            if match.start() != position:
                raise ValueError("fixture text is not token aligned")
            token_ids.append(int(match.group(1)))
            position = match.end()
        if position != len(text):
            raise ValueError("fixture text has an unmatched suffix")
        return tuple(token_ids)


def _source_value() -> dict[str, object]:
    return {
        "commit": _SOURCE_COMMIT,
        "expected_commit": _SOURCE_COMMIT,
        "expected_commit_matches": True,
        "git_clean": True,
        "git_status": [],
        "files": [
            {"path": path, "sha256": "a" * 64}
            for path in a7._SOURCE_PATHS
        ],
    }


def _mount(
    *,
    source: str,
    destination: str,
    mount_type: str = "bind",
    name: str | None = None,
) -> dict[str, object]:
    return {
        "type": mount_type,
        "name": name,
        "source": source,
        "destination": destination,
        "rw": False,
    }


def _container_core(
    *,
    container_id: str,
    service: str,
    include_model_mount: bool,
) -> dict[str, object]:
    return {
        "container_id": container_id,
        "service": service,
        "config_image": "fixture-image:latest",
        "image_id": _IMAGE_ID,
        "source_revision": _SOURCE_COMMIT,
        "working_dir": (
            "/workspace"
            if service == a7._GATEWAY_SERVICE
            else "/app"
        ),
        "running": True,
        "source_mount": _mount(
            source="repo:kairyu",
            destination="/app/kairyu",
        ),
        "model_mount": (
            _mount(
                source="volume:fixture-model",
                destination="/models",
                mount_type="volume",
                name="fixture-model",
            )
            if include_model_mount
            else None
        ),
    }


def _checkpoint_evidence() -> dict[str, object]:
    return a7.hashed_descriptor(
        {
            "schema_version": 1,
            "model": a7.MODEL,
            "checkpoint_root": a7._CHECKPOINT_ROOT,
            "model_revision": a7.MODEL_REVISION,
            "checkpoint": a7._expected_checkpoint_identity(),
            "revision_files": [
                {
                    "file": (
                        ".cache/huggingface/download/config.json.metadata"
                    ),
                    "revision": a7.MODEL_REVISION,
                }
            ],
            "container": _container_core(
                container_id="c" * 64,
                service=a7._ENGINE_SERVICE,
                include_model_mount=True,
            ),
            "reference_artifacts": a7._checkpoint_references(),
        }
    )


def _engine_config(tensor_parallel_size: int) -> str:
    return (
        a7._REPO_ROOT
        / "bench/deploy/qwen3-32b-multi-gpu/kairyu.template.yaml"
    ).read_text(encoding="utf-8").replace(
        "__TENSOR_PARALLEL_SIZE__",
        str(tensor_parallel_size),
    )


def _runtime_value(
    tensor_parallel_size: int,
    path: str,
) -> dict[str, object]:
    engine = {
        "container": _container_core(
            container_id=f"{tensor_parallel_size}" * 64,
            service=a7._ENGINE_SERVICE,
            include_model_mount=True,
        ),
        "template_mount": _mount(
            source=(
                "repo:bench/deploy/qwen3-32b-multi-gpu/"
                "kairyu.template.yaml"
            ),
            destination="/etc/kairyu/config.template.yaml",
        ),
        "port_binding": {
            "container_port": "8000/tcp",
            "host_ip": "127.0.0.1",
            "host_port": "8001",
        },
        "device_request": {
            "driver": "nvidia",
            "count": 0,
            "device_ids": [
                f"GPU-{index}"
                for index in range(tensor_parallel_size)
            ],
            "capabilities": [["gpu"]],
        },
        "gpu_inventory": [
            f"{index}, GPU-{index}"
            for index in range(tensor_parallel_size)
        ],
        "pid1_cmdline": [
            "/app/.venv/bin/kairyu",
            "serve",
            a7._ENGINE_CONFIG_PATH,
        ],
        "effective_config": a7.hashed_descriptor(
            _engine_config(tensor_parallel_size)
        ),
    }
    gateway = None
    if path == "gateway":
        gateway = {
            "container": _container_core(
                container_id=("d" if tensor_parallel_size == 4 else "e") * 64,
                service=a7._GATEWAY_SERVICE,
                include_model_mount=False,
            ),
            "workspace_mount": _mount(
                source="repo:.",
                destination="/workspace",
            ),
            "all_mounts": [
                _mount(
                    source="repo:kairyu",
                    destination="/app/kairyu",
                ),
                _mount(
                    source="repo:.",
                    destination="/workspace",
                ),
            ],
            "network_mode": "host",
            "pid1_cmdline": [
                "/app/.venv/bin/kairyu",
                "serve",
                "bench/deploy/qwen3-32b-multi-gpu/auto-gateway.yaml",
            ],
            "effective_config": a7.hashed_descriptor(
                (
                    a7._REPO_ROOT
                    / "bench/deploy/qwen3-32b-multi-gpu/auto-gateway.yaml"
                ).read_text(encoding="utf-8")
            ),
        }
    return {
        "schema_version": 1,
        "tensor_parallel_size": tensor_parallel_size,
        "path": path,
        "checkpoint_evidence_sha256": _checkpoint_evidence()["sha256"],
        "engine": engine,
        "gateway": gateway,
    }


def _configuration_value(
    tensor_parallel_size: int,
    path: str,
) -> dict[str, object]:
    return {
        "scenario": {
            "tensor_parallel_size": tensor_parallel_size,
            "path": path,
            "endpoint": (
                "http://127.0.0.1:8001"
                if path == "direct"
                else "http://127.0.0.1:8002"
            ),
        },
        "model": a7.MODEL,
        "model_revision": a7.MODEL_REVISION,
        "model_digests": {
            "weights-rollup": a7.MODEL_WEIGHTS_ROLLUP,
        },
        "tokenizer_sha256": _TOKENIZER_SHA256,
        "config_files": [
            {
                "path": "bench/deploy/qwen3-32b-multi-gpu/compose.yaml",
                "sha256": "d" * 64,
            },
            {
                "path": "bench/deploy/qwen3-32b-multi-gpu/auto-gateway.yaml",
                "sha256": "e" * 64,
            },
        ],
        "checkpoint_evidence": _checkpoint_evidence(),
    }


def _environment_value() -> dict[str, object]:
    return {
        "platform": "fixture-linux",
        "python": "fixture-python",
        "python_executable": "/fixture/python",
        "cuda_visible_devices": None,
        "nvidia_smi_gpus": [
            f"{index}, Fixture GPU, 96 GiB, 0000:{index:02x}:00.0, GPU-{index}"
            for index in range(8)
        ],
        "nvidia_smi_topology": [
            "GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7 CPU Affinity",
            *[
                f"GPU{index} X PIX PIX PIX PIX PIX PIX PIX 0-15"
                for index in range(8)
            ],
        ],
    }


def _provenance(
    tensor_parallel_size: int,
    path: str,
) -> dict[str, object]:
    return {
        "source": a7.hashed_descriptor(_source_value()),
        "configuration": a7.hashed_descriptor(
            _configuration_value(tensor_parallel_size, path)
        ),
        "environment": a7.hashed_descriptor(_environment_value()),
        "runtime": a7.hashed_descriptor(
            _runtime_value(tensor_parallel_size, path)
        ),
    }


def _backends(
    tensor_parallel_size: int,
    path: str,
) -> dict[str, object]:
    if path == "direct":
        return {
            "role": "engine-host",
            "engines": [
                {
                    "model": a7.MODEL,
                    "engine_backend": "kairyu",
                    "tensor_parallel_size": tensor_parallel_size,
                }
            ],
        }
    return {
        "role": "gateway",
        "engines": [
            {
                "model": a7.MODEL,
                "engine_backend": "replica-pool",
                "tensor_parallel_size": tensor_parallel_size,
                "via_replica": {
                    "tensor_parallel_size": tensor_parallel_size,
                },
            }
        ],
    }


def _measurement_rows(
    tensor_parallel_size: int,
    path: str,
    *,
    cached_tokens: int = 90,
) -> list[dict[str, object]]:
    prompts = a7.build_text_workload(_StablePieceTokenizer(), path=path)
    trace = a7.trace_descriptor(
        prompts,
        path=path,
        tokenizer_sha256=_TOKENIZER_SHA256,
    )
    affinity_before = (
        {
            "http_status": 200,
            "relevant_lines": [f"{a7._AFFINITY_METRIC_PREFIX}10.0"],
            "value": 10.0,
            "error_type": None,
        }
        if path == "gateway"
        else None
    )
    affinity_after = (
        {
            "http_status": 200,
            "relevant_lines": [f"{a7._AFFINITY_METRIC_PREFIX}458.0"],
            "value": 458.0,
            "error_type": None,
        }
        if path == "gateway"
        else None
    )
    scenario_id = a7._scenario_id(tensor_parallel_size, path)
    provenance = _provenance(tensor_parallel_size, path)
    rows: list[dict[str, object]] = [
        {
            "type": "run",
            "schema_version": a7.SCHEMA_VERSION,
            "gate": a7.GATE,
            "scenario_id": scenario_id,
            "tensor_parallel_size": tensor_parallel_size,
            "path": path,
            "workload": a7.workload_config(),
            "trace": trace,
            "backends": _backends(tensor_parallel_size, path),
            "provenance_start": provenance,
            "gateway_session_affinity_before": affinity_before,
        }
    ]
    trace_rows = trace["value"]["rows"]
    assert isinstance(trace_rows, list)
    for trace_row in trace_rows:
        assert isinstance(trace_row, dict)
        rows.append(
            {
                "type": "request",
                "scenario_id": scenario_id,
                "tensor_parallel_size": tensor_parallel_size,
                "path": path,
                **trace_row,
                "response_request_id": (
                    f"request-{tensor_parallel_size}-{path}-"
                    f"{trace_row['sequence']}"
                ),
                "http_status": 200,
                "status": "success",
                "error_type": None,
                "usage": {
                    "prompt_tokens": 100,
                    "cached_tokens": cached_tokens,
                    "completion_tokens": 1,
                },
                # A routing proxy may be retained as a diagnostic by callers,
                # but the A7 evaluator must never consult it.
                "router_cache_hit": False,
            }
        )
    rows.append(
        {
            "type": "run_end",
            "scenario_id": scenario_id,
            "tensor_parallel_size": tensor_parallel_size,
            "path": path,
            "request_count": a7.EXPECTED_REQUESTS,
            "provenance_end": provenance,
            "gateway_session_affinity_after": affinity_after,
            "gateway_session_affinity_delta": (
                a7._affinity_metric_delta(affinity_before, affinity_after)
            ),
        }
    )
    return rows


def _write_shards(
    tmp_path: Path,
    *,
    cached_tokens: int = 90,
) -> list[Path]:
    paths = []
    for tensor_parallel_size, path in a7.EXPECTED_SCENARIOS:
        shard = tmp_path / f"tp{tensor_parallel_size}-{path}.jsonl"
        a7.write_measurement_shard(
            shard,
            _measurement_rows(
                tensor_parallel_size,
                path,
                cached_tokens=cached_tokens,
            ),
        )
        paths.append(shard)
    return paths


def _passing_artifact(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    rows = a7.assemble_rows(_write_shards(tmp_path))
    artifact = tmp_path / "artifact"
    return artifact, a7.write_artifact(artifact, rows)


def test_qwen_text_replacement_preserves_fixed_geometry_and_theoretical_rate() -> None:
    tokenizer = _StablePieceTokenizer()
    direct = a7.build_text_workload(tokenizer, path="direct")
    gateway = a7.build_text_workload(tokenizer, path="gateway")
    geometry = a7._fixed_geometry()

    assert len(direct) == len(gateway) == a7.EXPECTED_REQUESTS == 512
    assert [item.sequence for item in direct] == list(range(512))
    assert [
        (item.session, item.turn, len(item.prompt_token_ids))
        for item in direct
    ] == [
        (row["session"], row["turn"], row["prompt_tokens"])
        for row in geometry
    ]
    assert sum(row["prompt_tokens"] for row in geometry) == 557_056
    assert sum(row["expected_cached_tokens"] for row in geometry) == 491_008
    assert 491_008 / 557_056 == pytest.approx(0.8814338235294118)
    assert direct[0].prompt_token_ids[0] != gateway[0].prompt_token_ids[0]

    direct_trace = a7.trace_descriptor(
        direct,
        path="direct",
        tokenizer_sha256=_TOKENIZER_SHA256,
    )
    assert a7._trace_valid(direct_trace, path="direct")


def test_artifact_round_trip_uses_only_engine_usage_for_the_binding_rate(
    tmp_path: Path,
) -> None:
    artifact, manifest = _passing_artifact(tmp_path)

    assert a7._usage_from_response(
        {
            "prompt_tokens": 100,
            "completion_tokens": 1,
            "prompt_tokens_details": None,
        }
    ) == {
        "prompt_tokens": 100,
        "cached_tokens": 0,
        "completion_tokens": 1,
    }
    assert a7._usage_valid(
        {
            "prompt_tokens": 100,
            "cached_tokens": 90,
            "completion_tokens": 0,
        }
    )
    assert manifest["passed"] is True
    assert all(manifest["checks"].values())
    assert {
        item["pooled_engine_cache_rate"]
        for item in manifest["scenarios"].values()
    } == {0.9}
    assert {
        key: manifest["diagnostics"][key]
        for key in (
            "latency_is_binding",
            "output_equality_is_binding",
            "routing_proxy_is_binding",
            "affinity_is_binding",
            "repeat_count_is_binding",
        )
    } == {
        "latency_is_binding": False,
        "output_equality_is_binding": False,
        "routing_proxy_is_binding": False,
        "affinity_is_binding": False,
        "repeat_count_is_binding": False,
    }
    assert manifest["diagnostics"]["gateway_session_affinity_by_scenario"][
        "tp8-gateway"
    ]["delta"] == 448.0
    assert a7.verify_artifact(artifact, assert_gate=True) == manifest


@pytest.mark.parametrize(
    "mutation,failed_check",
    [
        (
            "cache_floor",
            "every_pooled_engine_cache_rate_strictly_above_0_80",
        ),
        ("missing_request", "every_scenario_has_512_requests"),
        ("wrong_topology", "backends_topology_exact"),
        (
            "changed_provenance",
            "provenance_start_end_and_cross_scenario_exact",
        ),
    ],
)
def test_binding_failures_produce_a_replayable_rejected_gate(
    tmp_path: Path,
    mutation: str,
    failed_check: str,
) -> None:
    rows = a7.assemble_rows(_write_shards(tmp_path))
    mutated = copy.deepcopy(rows)
    if mutation == "cache_floor":
        for row in mutated:
            if (
                row.get("type") == "request"
                and row.get("scenario_id") == "tp4-direct"
            ):
                row["usage"]["cached_tokens"] = 80
    elif mutation == "missing_request":
        for index, row in enumerate(mutated):
            if (
                row.get("type") == "request"
                and row.get("scenario_id") == "tp4-direct"
            ):
                del mutated[index]
                break
    elif mutation == "wrong_topology":
        start = next(
            row
            for row in mutated
            if row.get("type") == "scenario_start"
            and row.get("scenario_id") == "tp8-gateway"
        )
        start["backends"]["engines"][0]["via_replica"][
            "tensor_parallel_size"
        ] = 4
    else:
        end = next(
            row
            for row in mutated
            if row.get("type") == "scenario_end"
            and row.get("scenario_id") == "tp4-direct"
        )
        changed = copy.deepcopy(end["provenance_end"]["environment"]["value"])
        changed["platform"] = "different-platform"
        end["provenance_end"]["environment"] = a7.hashed_descriptor(changed)

    artifact = tmp_path / f"artifact-{mutation}"
    manifest = a7.write_artifact(artifact, mutated)
    assert manifest["passed"] is False
    assert manifest["checks"][failed_check] is False
    assert a7.verify_artifact(artifact)["passed"] is False
    with pytest.raises(RuntimeError, match="does not pass"):
        a7.verify_artifact(artifact, assert_gate=True)


def test_raw_byte_tampering_is_rejected_before_semantic_replay(
    tmp_path: Path,
) -> None:
    artifact, _ = _passing_artifact(tmp_path)
    raw = artifact / a7.RAW_NAME
    original = raw.read_text()
    raw.write_text(
        original.replace('"cached_tokens":90', '"cached_tokens":89', 1)
    )

    with pytest.raises(ValueError, match="SHA-256"):
        a7.verify_artifact(artifact)


def test_manifest_tampering_is_rejected_even_when_raw_is_unchanged(
    tmp_path: Path,
) -> None:
    artifact, manifest = _passing_artifact(tmp_path)
    manifest["passed"] = False
    (artifact / a7.MANIFEST_NAME).write_text(
        a7.canonical_json(manifest) + "\n"
    )

    with pytest.raises(ValueError, match="independent raw replay"):
        a7.verify_artifact(artifact)
